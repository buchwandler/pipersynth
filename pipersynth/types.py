from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .audio import audio_to_int16_bytes, float_to_int16, write_wav
from .diagnostics import RuntimeDiagnostics
from .errors import InvalidSynthesisConfigError, ModelInferenceError
from .voice_level import VoiceLevelConfig


def _finite(value: float, name: str, *, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSynthesisConfigError(f"{name} must be a finite number")
    if not np.isfinite(value) or value < minimum:
        raise InvalidSynthesisConfigError(f"{name} must be finite and >= {minimum}")


def _valid_span(start: int, end: int) -> None:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or start >= end
    ):
        raise ValueError("span must satisfy 0 <= start < end")


def _validated_audio(audio: np.ndarray, *, name: str) -> np.ndarray:
    result = np.array(audio, dtype=np.float32, copy=True)
    if result.ndim != 1:
        raise ModelInferenceError(f"{name} must be mono, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ModelInferenceError(f"{name} contains non-finite samples")
    return result


def _validate_rate(sample_rate: int) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    start: int
    end: int
    phonemes: str | None = None
    language: str | None = None
    stress: str | None = None

    def __post_init__(self) -> None:
        _valid_span(self.start, self.end)
        if all(value is None for value in (self.phonemes, self.language, self.stress)):
            raise ValueError("pronunciation override must specify at least one effect")
        for name, value in (
            ("phonemes", self.phonemes),
            ("language", self.language),
            ("stress", self.stress),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.stress is not None and self.stress not in {"-2", "-1", "1", "2"}:
            raise ValueError("stress must be one of '-2', '-1', '1', or '2'")


@dataclass(frozen=True, slots=True)
class LinguisticToken:
    start: int
    end: int
    text: str | None = None
    pos: str | None = None
    tag: str | None = None
    lemma: str | None = None
    language: str | None = None
    morph: str | None = None

    def __post_init__(self) -> None:
        _valid_span(self.start, self.end)
        for name in ("text", "pos", "tag", "lemma", "language", "morph"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")


@dataclass(frozen=True, slots=True)
class SynthesisSegment:
    id: str
    text: str
    language: str
    speaker: int | str | None = None
    pronunciation_overrides: tuple[PronunciationOverride, ...] = ()
    annotations: tuple[LinguisticToken, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string")
        if self.speaker is not None and (
            isinstance(self.speaker, bool) or not isinstance(self.speaker, (int, str))
        ):
            raise ValueError("speaker must be an integer, name, or None")
        if isinstance(self.speaker, str) and not self.speaker:
            raise ValueError("speaker name must be non-empty")
        overrides = tuple(self.pronunciation_overrides)
        annotations = tuple(self.annotations)
        if any(not isinstance(item, PronunciationOverride) for item in overrides):
            raise TypeError("pronunciation_overrides must contain PronunciationOverride values")
        if any(not isinstance(item, LinguisticToken) for item in annotations):
            raise TypeError("annotations must contain LinguisticToken values")
        for override in overrides:
            if override.end > len(self.text):
                raise ValueError("pronunciation override span exceeds text length")
        for annotation in annotations:
            if annotation.end > len(self.text):
                raise ValueError("annotation span exceeds text length")
            if (
                annotation.text is not None
                and self.text[annotation.start : annotation.end] != annotation.text
            ):
                raise ValueError("annotation text does not match its source span")
        object.__setattr__(self, "pronunciation_overrides", overrides)
        object.__setattr__(self, "annotations", annotations)


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    """Per-request Piper inference and engine-local output controls."""

    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    normalize_audio: bool = True
    output_gain: float = 1.0
    voice_level: VoiceLevelConfig = field(default_factory=VoiceLevelConfig)

    def __post_init__(self) -> None:
        if self.length_scale is not None:
            _finite(self.length_scale, "length_scale", minimum=np.finfo(float).tiny)
        if self.noise_scale is not None:
            _finite(self.noise_scale, "noise_scale", minimum=0.0)
        if self.noise_w_scale is not None:
            _finite(self.noise_w_scale, "noise_w_scale", minimum=0.0)
        if not isinstance(self.normalize_audio, bool):
            raise InvalidSynthesisConfigError("normalize_audio must be a bool")
        _finite(self.output_gain, "output_gain", minimum=0.0)
        if not isinstance(self.voice_level, VoiceLevelConfig):
            raise InvalidSynthesisConfigError("voice_level must be a VoiceLevelConfig")


@dataclass(slots=True)
class RenderedChunk:
    index: int
    audio: np.ndarray
    sample_rate: int
    segment_id: str
    phonemes: tuple[str, ...] = ()
    phoneme_ids: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError("segment_id must be a non-empty string")
        _validate_rate(self.sample_rate)
        self.audio = _validated_audio(self.audio, name="chunk audio")
        self.phonemes = tuple(self.phonemes)
        self.phoneme_ids = tuple(self.phoneme_ids)
        self.warnings = tuple(self.warnings)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RenderedSegment:
    id: str
    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    speaker_id: int | None
    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    warnings: tuple[str, ...] = ()
    chunks: tuple[RenderedChunk, ...] = ()
    diagnostics: RuntimeDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string")
        if self.speaker_id is not None and (
            isinstance(self.speaker_id, bool)
            or not isinstance(self.speaker_id, int)
            or self.speaker_id < 0
        ):
            raise ValueError("speaker_id must be a non-negative integer or None")
        _validate_rate(self.sample_rate)
        self.audio = _validated_audio(self.audio, name="result audio")
        self.phonemes = tuple(self.phonemes)
        self.phoneme_ids = tuple(self.phoneme_ids)
        self.warnings = tuple(self.warnings)
        self.chunks = tuple(self.chunks)
        self.metadata = dict(self.metadata)
        if any(not isinstance(chunk, RenderedChunk) for chunk in self.chunks):
            raise TypeError("chunks must contain RenderedChunk values")

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def int16_array(self) -> np.ndarray:
        return float_to_int16(self.audio)

    @property
    def int16_bytes(self) -> bytes:
        return audio_to_int16_bytes(self.audio)

    def save_wav(self, target: str | Path | BinaryIO) -> str | Path | BinaryIO:
        write_wav(target, self.audio, self.sample_rate)
        return target
