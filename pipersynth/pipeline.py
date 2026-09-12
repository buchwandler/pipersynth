from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from piperg2p import PhonemeSentence, PhonemizeResult

from .audio import silence_samples
from .config import GenerationConfig, PipelineConfig
from .diagnostics import TimingDiagnostics
from .errors import OptionalDependencyError, VoiceClosedError
from .preparation import IdentityTextPreparer, PreparedTextResult, SpokenformTextPreparer
from .types import AudioChunk, AudioResult, AudioUnitDescriptor, AudioUnitResult, SynthesisConfig
from .voice import PiperVoice


class PreparedAudioUnits:
    """Prepared sentence units rendered on demand by a reusable pipeline."""

    def __init__(
        self,
        pipeline: PiperPipeline,
        prepared_text: PreparedTextResult,
        frontend_result: Any,
        generation: GenerationConfig,
        unit_kind: Literal["sentence", "paragraph"],
        unit_texts: tuple[str, ...] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._prepared_text = prepared_text
        self._frontend_result = frontend_result
        self._generation = generation
        self._closed = False
        self.unit_kind = unit_kind
        self.units = tuple(
            AudioUnitDescriptor(
                index,
                unit_kind,
                (unit_texts[index] if unit_texts is not None else sentence.phoneme_string),
            )
            for index, sentence in enumerate(frontend_result.sentences)
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise VoiceClosedError("PreparedAudioUnits is closed")

    def render(
        self,
        indices: tuple[int, ...] | list[int] | None = None,
        skip_indices: tuple[int, ...] | list[int] = (),
    ) -> Iterator[AudioUnitResult]:
        """Render selected units lazily, preserving their stable source indices."""

        self._ensure_open()
        selected = tuple(range(len(self.units))) if indices is None else tuple(indices)
        skipped = set(skip_indices)
        for index in selected:
            if index in skipped:
                continue
            if index < 0 or index >= len(self.units):
                raise IndexError(f"unit index {index} is out of range")
            sentence = self._frontend_result.sentences[index]
            if not sentence.ids:
                continue
            voice = self._pipeline.voice
            audio = voice.synthesize_ids(
                sentence.ids,
                self._pipeline._to_synthesis_config(self._generation),
            )
            metadata: dict[str, Any] = {}
            if self._frontend_result.diagnostics is not None:
                metadata["frontend_diagnostics"] = self._frontend_result.diagnostics
            yield AudioUnitResult(
                descriptor=self.units[index],
                audio=audio,
                sample_rate=voice.config.sample_rate,
                phonemes=tuple(sentence.phonemes),
                phoneme_ids=tuple(sentence.ids),
                warnings=tuple(sentence.warnings),
                metadata=metadata,
            )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> PreparedAudioUnits:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PiperPipeline:
    """Reusable high-level text preparation and Piper voice synthesis pipeline."""

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
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        frontend_options: Mapping[str, Any] | None = None,
        text_preparation: Literal["identity", "spokenform"] = "identity",
        language: str | None = None,
        retain_unit_audio: bool = False,
        return_diagnostics: bool = True,
        progress: Callable[..., Any] | None = None,
    ) -> PiperPipeline:
        from .asset_manager import VoiceAssetManager
        from .preparation import normalize_catalog_language_for_spokenform

        manager = VoiceAssetManager(cache_dir, offline=offline, progress=progress)
        bundle = manager.resolve_voice(
            voice, refresh_catalog=refresh_catalog, force_download=force_download
        )
        if text_preparation == "spokenform" and language is None:
            if bundle.metadata is None:
                raise ValueError("spokenform language cannot be inferred without voice metadata")
            language = normalize_catalog_language_for_spokenform(bundle.metadata.language_code)
        config = PipelineConfig(
            model_path=bundle.model_path,
            config_path=bundle.config_path,
            generation=generation or GenerationConfig(),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            frontend_options=frontend_options,
            text_preparation=text_preparation,
            language=language,
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
        text_preparer: Callable[[str, str | None], PreparedTextResult] | None = None,
    ) -> None:
        self.config = config
        self._voice_factory = voice_factory
        self._text_preparer = text_preparer
        self._default_text_preparer = (
            SpokenformTextPreparer()
            if config.text_preparation == "spokenform"
            else IdentityTextPreparer()
        )
        self._voice: PiperVoice | None = None
        self._voice_bundle: Any = None
        self._last_timing: dict[str, float] = {}
        self._prepared: list[PreparedAudioUnits] = []
        self._closed = False

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
        """Return managed asset provenance, if this is a pretrained pipeline."""

        return self._voice_bundle

    def _prepare_text(self, text: str) -> PreparedTextResult:
        if self._text_preparer is not None:
            return self._text_preparer(text, self.config.language)
        return self._default_text_preparer.prepare(text, language=self.config.language)

    def _generation(self, overrides: Mapping[str, Any]) -> GenerationConfig:
        unknown = set(overrides) - set(GenerationConfig.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown generation override(s): {names}")
        return replace(self.config.generation, **dict(overrides))

    def _to_synthesis_config(self, generation: GenerationConfig) -> SynthesisConfig:
        speaker_id = self.voice.resolve_speaker_id(generation.speaker)
        return SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=generation.length_scale,
            noise_scale=generation.noise_scale,
            noise_w_scale=generation.noise_w_scale,
            normalize_audio=generation.normalize_audio,
            volume=generation.volume,
        )

    def prepare_units(
        self,
        text: str,
        *,
        unit: Literal["sentence", "paragraph"] = "sentence",
        **overrides: Any,
    ) -> PreparedAudioUnits:
        self._ensure_open()
        generation = self._generation(overrides)
        prepared_started = time.perf_counter()
        prepared = self._prepare_text(text)
        prepared_ms = (time.perf_counter() - prepared_started) * 1000
        phonemize_started = time.perf_counter()
        if unit == "sentence":
            frontend_result = self.voice.frontend.phonemize_prepared(prepared.prepared_text)
            unit_texts = None
        elif unit == "paragraph":
            paragraphs = tuple(
                part for part in prepared.prepared_text.split("\n\n") if part.strip()
            )
            grouped: list[PhonemeSentence] = []
            diagnostics = None
            for paragraph in paragraphs:
                paragraph_result = self.voice.frontend.phonemize_prepared(paragraph)
                diagnostics = paragraph_result.diagnostics
                sentences = paragraph_result.sentences
                if sentences:
                    grouped.append(
                        PhonemeSentence(
                            tuple(phone for sentence in sentences for phone in sentence.phonemes),
                            tuple(identifier for sentence in sentences for identifier in sentence.ids),
                            warnings=tuple(
                                warning for sentence in sentences for warning in sentence.warnings
                            ),
                        )
                    )
            frontend_result = PhonemizeResult(prepared.prepared_text, tuple(grouped), diagnostics)
            unit_texts = paragraphs
        else:
            raise ValueError("unit must be 'sentence' or 'paragraph'")
        self._last_timing = {
            "prepare_text_ms": prepared_ms,
            "phonemize_ms": (time.perf_counter() - phonemize_started) * 1000,
        }
        result = PreparedAudioUnits(
            self, prepared, frontend_result, generation, unit, unit_texts
        )
        self._prepared.append(result)
        return result

    def iter_units(
        self,
        text: str,
        *,
        unit: Literal["sentence", "paragraph"] = "sentence",
        **overrides: Any,
    ) -> Iterator[AudioUnitResult]:
        prepared = self.prepare_units(text, unit=unit, **overrides)
        try:
            yield from prepared.render()
        finally:
            prepared.close()

    def iter_pcm(
        self,
        text: str,
        *,
        unit: Literal["sentence", "paragraph"] = "sentence",
        **overrides: Any,
    ) -> Iterator[bytes]:
        """Yield mono 16-bit PCM bytes for each prepared audio unit."""

        for unit_result in self.iter_units(text, unit=unit, **overrides):
            yield unit_result.audio_int16_bytes

    def play_streaming(
        self,
        text: str,
        *,
        unit: Literal["sentence", "paragraph"] = "sentence",
        **overrides: Any,
    ) -> None:
        """Play synthesized units through the optional sounddevice dependency."""

        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Audio playback requires sounddevice. Install pipersynth[playback]."
            ) from exc
        for unit_result in self.iter_units(text, unit=unit, **overrides):
            sd.play(unit_result.audio, unit_result.sample_rate, blocking=True)


    def run(self, text: str, **overrides: Any) -> AudioResult:
        self._ensure_open()
        run_started = time.perf_counter()
        generation = self._generation(overrides)
        prepared = self.prepare_units(text, **overrides)
        inference_started = time.perf_counter()
        try:
            units = list(prepared.render())
        finally:
            prepared.close()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        postprocess_started = time.perf_counter()
        silence_count = silence_samples(self.voice.config.sample_rate, generation.sentence_silence)
        if not units:
            audio = np.zeros(0, dtype=np.float32)
        else:
            parts: list[np.ndarray] = []
            silence = np.zeros(silence_count, dtype=np.float32)
            for index, unit_result in enumerate(units):
                if index and silence_count:
                    parts.append(silence)
                parts.append(unit_result.audio)
            audio = np.concatenate(parts).astype(np.float32, copy=False)
        chunks = [
            AudioChunk(
                sample_rate=unit.sample_rate,
                audio_float_array=unit.audio,
                phonemes=unit.phonemes,
                phoneme_ids=unit.phoneme_ids,
                warnings=unit.warnings,
                metadata=unit.metadata,
            )
            for unit in units
        ]
        warnings = tuple(prepared._prepared_text.warnings) + tuple(
            warning for unit in units for warning in unit.warnings
        )
        timing = TimingDiagnostics(
            prepare_text_ms=self._last_timing.get("prepare_text_ms"),
            phonemize_ms=self._last_timing.get("phonemize_ms"),
            inference_ms=inference_ms,
            postprocess_ms=(time.perf_counter() - postprocess_started) * 1000,
            total_ms=(time.perf_counter() - run_started) * 1000,
        )
        return AudioResult(
            audio=audio,
            sample_rate=self.voice.config.sample_rate,
            source_text=text,
            prepared_text=prepared._prepared_text.prepared_text,
            chunks=chunks if self.config.retain_unit_audio else [],
            warnings=warnings,
            diagnostics=self.voice.diagnostics if self.config.return_diagnostics else None,
            timing=timing if self.config.return_diagnostics else None,
            metadata=dict(prepared._prepared_text.metadata),
        )

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
