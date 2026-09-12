from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from .audio import silence_samples
from .config import GenerationConfig, PipelineConfig
from .errors import VoiceClosedError
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
    ) -> None:
        self._pipeline = pipeline
        self._prepared_text = prepared_text
        self._frontend_result = frontend_result
        self._generation = generation
        self._closed = False
        self.unit_kind = unit_kind
        self.units = tuple(
            AudioUnitDescriptor(index, unit_kind, sentence.phoneme_string)
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
        if unit != "sentence":
            raise ValueError("only sentence units are currently supported")
        prepared = self._prepare_text(text)
        frontend_result = self.voice.frontend.phonemize_prepared(prepared.prepared_text)
        result = PreparedAudioUnits(self, prepared, frontend_result, self._generation(overrides), unit)
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

    def run(self, text: str, **overrides: Any) -> AudioResult:
        self._ensure_open()
        generation = self._generation(overrides)
        prepared = self.prepare_units(text, **overrides)
        try:
            units = list(prepared.render())
        finally:
            prepared.close()
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
        return AudioResult(
            audio=audio,
            sample_rate=self.voice.config.sample_rate,
            source_text=text,
            prepared_text=prepared._prepared_text.prepared_text,
            chunks=chunks if self.config.retain_unit_audio else [],
            warnings=warnings,
            diagnostics=self.voice.diagnostics if self.config.return_diagnostics else None,
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
