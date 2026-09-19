from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from audiocompose import AudioJob, Composer
from utterplan import (
    LinguisticsConfig,
    PauseConfig,
    SSMDConfig,
    UtterancePlan,
    UtterancePlanner,
    normalize_language,
)

from .audio import silence_samples
from .audio_job import PiperAudioJobContext, build_audio_job_context, render_segment
from .composition import audio_result_from_composition
from .config import GenerationConfig, PipelineConfig
from .diagnostics import TimingDiagnostics
from .errors import ConfigFileNotFoundError, OptionalDependencyError, VoiceClosedError
from .loudness_config import LoudnessConfig, coerce_loudness
from .plan_adapter import PreparedPiperUnit
from .plan_adapter import prepare_plan as adapt_plan
from .planning import inspect_voice_config, planner_config_from_pipersynth
from .types import AudioResult, AudioUnitDescriptor, AudioUnitResult, SynthesisConfig
from .voice import PiperVoice


class PreparedAudioUnits:
    """Plan-backed audio units rendered lazily by a reusable pipeline."""

    def __init__(
        self,
        pipeline: PiperPipeline,
        plan: UtterancePlan,
        prepared_units: tuple[PreparedPiperUnit, ...],
        generation: GenerationConfig,
    ) -> None:
        self._pipeline = pipeline
        self.plan = plan
        self._prepared_units = prepared_units
        self._generation = generation
        self._closed = False
        self.unit_kind = plan.config.get("unit", "paragraph")
        self.units = tuple(
            AudioUnitDescriptor(
                index=unit.index,
                unit_kind=unit.kind,
                text=plan.texts.spoken[unit.spoken_start : unit.spoken_end],
                char_start=unit.spoken_start,
                char_end=unit.spoken_end,
                plan_unit_id=unit.plan_unit_id,
                content_hash=unit.content_hash,
                segment_ids=unit.segment_ids,
                marker_ids=unit.marker_ids,
            )
            for unit in prepared_units
        )
        self._by_index = {unit.index: unit for unit in prepared_units}
        self._descriptor_by_index = {descriptor.index: descriptor for descriptor in self.units}

    def _ensure_open(self) -> None:
        if self._closed:
            raise VoiceClosedError("PreparedAudioUnits is closed")

    def _marker_metadata(self, unit: PreparedPiperUnit, audio: np.ndarray) -> list[dict[str, Any]]:
        segment_by_id = {segment.id: segment for segment in self.plan.segments}
        offsets: dict[int, int] = {unit.spoken_start: 0, unit.spoken_end: audio.size}
        offset = 0
        for prepared in unit.segments:
            segment = segment_by_id[prepared.plan_segment_id]
            offset += silence_samples(
                self._pipeline.voice.config.sample_rate, prepared.pause_before_seconds
            )
            offsets[segment.spoken_start] = offset
            offset += self._segment_audio_sizes.get(prepared.plan_segment_id, 0)
            offsets[segment.spoken_end] = offset
            offset += silence_samples(
                self._pipeline.voice.config.sample_rate, prepared.pause_after_seconds
            )
        markers = {marker.id: marker for marker in self.plan.markers}
        result: list[dict[str, Any]] = []
        for marker_id in unit.marker_ids:
            marker = markers[marker_id]
            sample_offset = offsets.get(marker.spoken_position)
            result.append(
                {
                    "id": marker.id,
                    "name": marker.name,
                    "char_offset": marker.spoken_position,
                    "sample_offset": sample_offset,
                    "timing": "resolved" if sample_offset is not None else "unresolved",
                }
            )
        return result

    def _render_one(self, unit: PreparedPiperUnit) -> AudioUnitResult:
        voice = self._pipeline.voice
        parts: list[np.ndarray] = []
        phonemes: list[str] = []
        phoneme_ids: list[int] = []
        warnings: list[str] = []
        segment_by_id = {segment.id: segment for segment in self.plan.segments}
        self._segment_audio_sizes = {}
        rendered_metadata: list[Mapping[str, Any]] = []
        for prepared in unit.segments:
            plan_segment = segment_by_id[prepared.plan_segment_id]
            rendered = render_segment(
                prepared,
                unit_id=unit.plan_unit_id,
                spoken_start=plan_segment.spoken_start,
                spoken_end=plan_segment.spoken_end,
                voice=voice,
            )
            before = silence_samples(voice.config.sample_rate, rendered.pause_before_seconds)
            if before:
                parts.append(np.zeros(before, dtype=np.float32))
            self._segment_audio_sizes[rendered.segment_id] = rendered.audio.size
            if rendered.audio.size:
                parts.append(rendered.audio)
            after = silence_samples(voice.config.sample_rate, rendered.pause_after_seconds)
            if after:
                parts.append(np.zeros(after, dtype=np.float32))
            phonemes.extend(rendered.phonemes)
            phoneme_ids.extend(rendered.phoneme_ids)
            warnings.extend(rendered.warnings)
            rendered_metadata.append(rendered.metadata)
        audio = (
            np.concatenate(parts).astype(np.float32, copy=False)
            if parts
            else np.zeros(0, dtype=np.float32)
        )
        metadata = {
            "plan_id": self.plan.plan_id,
            "frontend_diagnostics": rendered_metadata,
        }
        metadata["markers"] = self._marker_metadata(unit, audio)
        return AudioUnitResult(
            descriptor=self._descriptor_by_index[unit.index],
            audio=audio,
            sample_rate=voice.config.sample_rate,
            phonemes=tuple(phonemes),
            phoneme_ids=tuple(phoneme_ids),
            warnings=tuple(warnings),
            metadata=metadata,
            plan_unit_id=unit.plan_unit_id,
            segment_ids=unit.segment_ids,
            marker_ids=unit.marker_ids,
        )

    def render(
        self,
        indices: tuple[int, ...] | list[int] | None = None,
        skip_indices: tuple[int, ...] | list[int] = (),
    ) -> Iterator[AudioUnitResult]:
        self._ensure_open()
        selected = tuple(self._by_index) if indices is None else tuple(indices)
        skipped = set(skip_indices)
        for index in selected:
            if index in skipped:
                continue
            if index not in self._by_index:
                raise IndexError(f"unit index {index} is out of range")
            yield self._render_one(self._by_index[index])

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> PreparedAudioUnits:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PiperPipeline:
    """Reusable UtterPlan semantic planner and Piper voice renderer."""

    @classmethod
    def from_pretrained(
        cls,
        voice: str,
        *,
        cache_dir: str | Path | None = None,
        offline: bool | None = None,
        refresh_catalog: bool = False,
        force_download: bool = False,
        generation: GenerationConfig | None = None,
        providers: Any | None = None,
        loudness: LoudnessConfig | Mapping[str, object] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        frontend_options: Mapping[str, Any] | None = None,
        text_preparation: Literal["identity", "spokenform"] = "identity",
        language: str | None = None,
        document_format: Literal["plain", "ssmd"] = "plain",
        unit: Literal["paragraph", "sentence"] = "paragraph",
        pauses: PauseConfig | None = None,
        linguistics: LinguisticsConfig | None = None,
        ssmd: SSMDConfig | None = None,
        overlap_mode: Literal["snap", "strict"] = "snap",
        language_aliases: Mapping[str, str] | None = None,
        planner_diagnostics: bool = True,
        directive_policy: Literal["error", "warn", "ignore"] = "error",
        language_policy: Literal["strict", "allow"] = "strict",
        retain_unit_audio: bool = False,
        return_diagnostics: bool = True,
        progress: Callable[..., Any] | None = None,
    ) -> PiperPipeline:
        from .asset_manager import VoiceAssetManager

        manager = VoiceAssetManager(cache_dir, offline=offline, progress=progress)
        bundle = manager.resolve_voice(
            voice, refresh_catalog=refresh_catalog, force_download=force_download
        )
        if language is None:
            if bundle.metadata is None:
                raise ValueError("planner language cannot be inferred without voice metadata")
            language = normalize_language(bundle.metadata.language_code)
        config = PipelineConfig(
            model_path=bundle.model_path,
            config_path=bundle.config_path,
            generation=generation or GenerationConfig(),
            loudness=coerce_loudness(loudness),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            frontend_options=frontend_options,
            language=language,
            document_format=document_format,
            text_preparation=text_preparation,
            unit=unit,
            pauses=pauses or PauseConfig(),
            linguistics=linguistics or LinguisticsConfig(),
            ssmd=ssmd or SSMDConfig(),
            overlap_mode=overlap_mode,
            language_aliases=language_aliases or {},
            planner_diagnostics=planner_diagnostics,
            directive_policy=directive_policy,
            language_policy=language_policy,
            retain_unit_audio=retain_unit_audio,
            return_diagnostics=return_diagnostics,
        )
        pipeline = cls(config)
        pipeline._voice_bundle = bundle
        return pipeline

    def __init__(
        self,
        config: PipelineConfig,
        *,
        voice_factory: Callable[[PipelineConfig], PiperVoice] | None = None,
        planner: UtterancePlanner | None = None,
    ) -> None:
        self.config = config
        self._voice_factory = voice_factory
        self._voice: PiperVoice | None = None
        self._voice_bundle: Any = None
        self._voice_info = None
        self._planner = planner
        self._prepared: list[PreparedAudioUnits] = []
        self._last_timing: dict[str, float] = {}
        self._closed = False
        if self._planner is None and config.language:
            self._planner = UtterancePlanner(planner_config_from_pipersynth(config))
        elif self._planner is None:
            try:
                self._voice_info = inspect_voice_config(config)
            except ConfigFileNotFoundError:
                pass
            else:
                self._planner = UtterancePlanner(
                    planner_config_from_pipersynth(
                        config, language=self._voice_info.planner_language
                    )
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise VoiceClosedError("PiperPipeline is closed")

    @property
    def voice(self) -> PiperVoice:
        self._ensure_open()
        if self._voice is None:
            if self._voice_factory is not None:
                self._voice = self._voice_factory(self.config)
            else:
                if (
                    self._voice_bundle is not None
                    and getattr(self._voice_bundle, "installation", None) is not None
                ):
                    self._voice = PiperVoice.from_bundle(
                        self._voice_bundle,
                        providers=self.config.providers,
                        provider_options=self.config.provider_options,
                        session_options=self.config.session_options,
                        frontend_options=self.config.frontend_options,
                    )
                else:
                    self._voice = PiperVoice.load(
                        self.config.model_path,
                        self.config.config_path,
                        providers=self.config.providers,
                        provider_options=self.config.provider_options,
                        session_options=self.config.session_options,
                        frontend_options=self.config.frontend_options,
                    )
        return self._voice

    @property
    def voice_bundle(self) -> Any:
        return self._voice_bundle

    def _ensure_planner(self, config: PipelineConfig) -> UtterancePlanner:
        self._ensure_open()
        if self._planner is not None:
            return self._planner
        if config.language:
            language = config.language
        elif self._voice_info is not None:
            language = self._voice_info.planner_language
        elif self._voice_factory is not None:
            candidate = getattr(self.voice.config, "espeak_voice", None)
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("A planner language could not be resolved")
            language = candidate
        else:
            self._voice_info = inspect_voice_config(config)
            language = self._voice_info.planner_language
        planner_config = planner_config_from_pipersynth(config, language=language)
        self._planner = UtterancePlanner(planner_config)
        return self._planner

    def _resolve_run_config(self, overrides: Mapping[str, Any]) -> PipelineConfig:
        generation_fields = set(GenerationConfig.__dataclass_fields__)
        planner_fields = {
            "language",
            "document_format",
            "text_preparation",
            "unit",
            "pauses",
            "linguistics",
            "ssmd",
            "overlap_mode",
            "language_aliases",
            "planner_diagnostics",
            "directive_policy",
            "language_policy",
        }
        unknown = set(overrides) - generation_fields - planner_fields - {"loudness"}
        loudness = (
            coerce_loudness(overrides["loudness"])
            if "loudness" in overrides
            else self.config.loudness
        )
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown run override(s): {names}")
        generation = replace(
            self.config.generation,
            **{key: value for key, value in overrides.items() if key in generation_fields},
        )
        pipeline = {key: value for key, value in overrides.items() if key in planner_fields}
        return replace(self.config, generation=generation, loudness=loudness, **pipeline)

    def _planner_config(self, config: PipelineConfig) -> Any:
        planner = self._ensure_planner(config)
        language = config.language
        if language is None:
            language = (
                self._voice_info.planner_language
                if self._voice_info is not None
                else getattr(self.voice.config, "espeak_voice", None)
            )
        return planner, planner_config_from_pipersynth(config, language=language)

    def plan(
        self,
        text: str,
        *,
        unit: Literal["paragraph", "sentence"] | None = None,
        **planning_overrides: Any,
    ) -> UtterancePlan:
        config = self._resolve_run_config(
            {**planning_overrides, **({"unit": unit} if unit is not None else {})}
        )
        planner, planner_config = self._planner_config(config)
        return planner.plan(text, config=planner_config, unit=unit)

    def _to_synthesis_config(
        self, generation: GenerationConfig, loudness: LoudnessConfig | None = None
    ) -> SynthesisConfig:
        return SynthesisConfig(
            speaker_id=self.voice.resolve_speaker_id(generation.speaker),
            length_scale=generation.length_scale,
            noise_scale=generation.noise_scale,
            noise_w_scale=generation.noise_w_scale,
            normalize_audio=generation.normalize_audio,
            volume=generation.volume,
            loudness=loudness or LoudnessConfig(),
        )

    def prepare_plan(self, plan: UtterancePlan, **render_overrides: Any) -> PreparedAudioUnits:
        self._ensure_open()
        planner_fields = {
            "language",
            "document_format",
            "text_preparation",
            "unit",
            "pauses",
            "linguistics",
            "ssmd",
            "overlap_mode",
            "language_aliases",
            "planner_diagnostics",
            "directive_policy",
            "language_policy",
        }
        if planner_fields.intersection(render_overrides):
            names = ", ".join(sorted(planner_fields.intersection(render_overrides)))
            raise TypeError(f"planning override(s) are not allowed while rendering a plan: {names}")
        config = render_overrides.pop("config", None)
        effective = (
            config
            if isinstance(config, PipelineConfig)
            else self._resolve_run_config(render_overrides)
        )
        started = time.perf_counter()
        prepared = adapt_plan(
            plan,
            self.voice,
            effective.generation,
            loudness=effective.loudness,
            directive_policy=effective.directive_policy,
            language_policy=effective.language_policy,
            language_aliases=effective.language_aliases,
        )
        self._last_timing["g2p_ms"] = (time.perf_counter() - started) * 1000
        result = PreparedAudioUnits(self, plan, prepared, effective.generation)
        self._prepared.append(result)
        return result

    def _build_audio_job_context(
        self, plan: UtterancePlan, **render_overrides: Any
    ) -> PiperAudioJobContext:
        self._ensure_open()
        plan.validate()
        config = render_overrides.get("config")
        effective = (
            config
            if isinstance(config, PipelineConfig)
            else self._resolve_run_config(render_overrides)
        )
        prepared = self.prepare_plan(plan, **dict(render_overrides))
        try:
            return build_audio_job_context(
                plan=plan,
                prepared_units=prepared._prepared_units,
                voice=self.voice,
                voice_id=getattr(self._voice_bundle, "voice_id", None),
                loudness=effective.loudness,
            )
        finally:
            prepared.close()

    def to_audio_job(self, plan: UtterancePlan, **render_overrides: Any) -> AudioJob:
        """Build a generic AudioCompose job without composing it."""
        return self._build_audio_job_context(plan, **render_overrides).job

    def render_plan(self, plan: UtterancePlan, **render_overrides: Any) -> AudioResult:
        self._ensure_open()
        plan.validate()
        config = render_overrides.get("config")
        effective = (
            config
            if isinstance(config, PipelineConfig)
            else self._resolve_run_config(render_overrides)
        )
        started = time.perf_counter()
        context = self._build_audio_job_context(plan, **render_overrides)
        composition_started = time.perf_counter()
        composition = Composer().compose(context.job)
        composition_ms = (time.perf_counter() - composition_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        voice_id = context.voice_id
        voice_source_revision = getattr(
            getattr(self._voice_bundle, "metadata", None), "source_revision", None
        )
        diagnostics = None
        if self.config.return_diagnostics:
            diagnostics = replace(
                self.voice.diagnostics,
                plan_id=plan.plan_id,
                utterplan_producer=dict(plan.producer),
                utterplan_schema_version=plan.schema_version,
                voice_id=voice_id,
                voice_source_revision=voice_source_revision,
            )
        timing = TimingDiagnostics(
            planning_ms=self._last_timing.get("planning_ms"),
            g2p_ms=self._last_timing.get("g2p_ms"),
            inference_ms=(composition_started - started) * 1000,
            composition_ms=composition_ms,
            postprocess_ms=0.0,
            total_ms=total_ms,
        )
        result = audio_result_from_composition(
            plan=plan,
            context=context,
            composition=composition,
            diagnostics=diagnostics,
            timing=timing if self.config.return_diagnostics else None,
            retain_unit_audio=self.config.retain_unit_audio,
        )
        result.metadata.update(
            {
                "voice_source_revision": voice_source_revision,
                "frontend": getattr(
                    getattr(self.voice.frontend, "diagnostics", None), "backend", None
                ),
                "provider": self.voice.diagnostics.providers_active,
            }
        )
        application = getattr(self.voice, "last_voice_level_application", None)
        if application is not None:
            result.metadata.update(
                {
                    "voice_leveling_mode": effective.loudness.voice_leveling,
                    "voice_calibration_key": str(application.key) if application.key else None,
                    "voice_calibration_gain_db": application.gain_db,
                    "voice_calibration_source": application.source,
                    "voice_calibration_corpus": getattr(
                        getattr(self.voice, "_calibration_catalog", None), "corpus", None
                    ),
                }
            )
        return result

    def prepare_units(
        self,
        text: str,
        *,
        unit: Literal["paragraph", "sentence"] | None = None,
        **overrides: Any,
    ) -> PreparedAudioUnits:
        effective = self._resolve_run_config(
            {**overrides, **({"unit": unit} if unit is not None else {})}
        )
        if effective.loudness.target_lufs is not None:
            raise ValueError(
                "target_lufs requires complete output and is not supported for streaming"
            )
        plan = self.plan(
            text,
            unit=effective.unit,
            language=effective.language,
            document_format=effective.document_format,
            text_preparation=effective.text_preparation,
            pauses=effective.pauses,
            linguistics=effective.linguistics,
            ssmd=effective.ssmd,
            overlap_mode=effective.overlap_mode,
            language_aliases=effective.language_aliases,
            planner_diagnostics=effective.planner_diagnostics,
        )
        return self.prepare_plan(plan, config=effective)

    def iter_units(
        self,
        text: str,
        *,
        unit: Literal["paragraph", "sentence"] | None = None,
        **overrides: Any,
    ) -> Iterator[AudioUnitResult]:
        prepared = self.prepare_units(text, unit=unit, **overrides)
        try:
            yield from prepared.render()
        finally:
            prepared.close()

    def iter_pcm(self, text: str, **overrides: Any) -> Iterator[bytes]:
        for unit in self.iter_units(text, **overrides):
            yield unit.audio_int16_bytes

    def play_streaming(self, text: str, **overrides: Any) -> None:
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Audio playback requires sounddevice. Install pipersynth[playback]."
            ) from exc
        for unit in self.iter_units(text, **overrides):
            sd.play(unit.audio, unit.sample_rate, blocking=True)

    def _run_phoneme_input(self, phonemes: str, config: PipelineConfig) -> AudioResult:
        result = self.voice.frontend.phonemize_prepared(f"[[{phonemes}]]")
        sentences = tuple(getattr(result, "sentences", ()))
        if not sentences:
            sentences = tuple(
                item for item in getattr(result, "tokens", ()) if hasattr(item, "ids")
            )
        ids = tuple(identifier for sentence in sentences for identifier in sentence.ids)
        audio = self.voice.synthesize_ids(
            ids, self._to_synthesis_config(config.generation, config.loudness)
        )
        return AudioResult(
            audio=audio,
            sample_rate=self.voice.config.sample_rate,
            source_text=phonemes,
            prepared_text=phonemes,
            warnings=tuple(getattr(result, "warnings", ())),
            diagnostics=self.voice.diagnostics if config.return_diagnostics else None,
            metadata={"mode": "phonemes"},
        )

    def run(self, text: str, **overrides: Any) -> AudioResult:
        self._ensure_open()
        config = self._resolve_run_config(overrides)
        if config.generation.is_phonemes:
            return self._run_phoneme_input(text, config)
        started = time.perf_counter()
        plan = self.plan(
            text,
            unit=config.unit,
            language=config.language,
            document_format=config.document_format,
            text_preparation=config.text_preparation,
            pauses=config.pauses,
            linguistics=config.linguistics,
            ssmd=config.ssmd,
            overlap_mode=config.overlap_mode,
            language_aliases=config.language_aliases,
            planner_diagnostics=config.planner_diagnostics,
        )
        self._last_timing["planning_ms"] = (time.perf_counter() - started) * 1000
        return self.render_plan(plan, config=config)

    def __call__(self, text: str, **overrides: Any) -> AudioResult:
        return self.run(text, **overrides)

    def warmup(self) -> None:
        self.voice.warmup()

    def close(self) -> None:
        if self._closed:
            return
        for prepared in self._prepared:
            prepared.close()
        self._prepared.clear()
        if self._planner is not None:
            self._planner.close()
        if self._voice is not None:
            self._voice.close()
        self._closed = True

    def __enter__(self) -> PiperPipeline:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_pipeline(config: PipelineConfig, **kwargs: Any) -> PiperPipeline:
    return PiperPipeline(config, **kwargs)
