from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

from .audio import audio_to_int16_bytes, float_to_int16, write_wav
from .diagnostics import RuntimeDiagnostics, TimingDiagnostics
from .errors import InvalidSynthesisConfigError, ModelInferenceError, OptionalDependencyError


def _finite(value: float, name: str, *, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSynthesisConfigError(f"{name} must be a finite number")
    if not np.isfinite(value) or value < minimum:
        raise InvalidSynthesisConfigError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    """Per-call acoustic inference and output controls."""

    speaker_id: int | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    normalize_audio: bool = True
    volume: float = 1.0
    noise_w: float | None = None

    def __post_init__(self) -> None:
        if self.speaker_id is not None and (
            isinstance(self.speaker_id, bool) or not isinstance(self.speaker_id, int)
        ):
            raise InvalidSynthesisConfigError("speaker_id must be an integer or None")
        if self.length_scale is not None:
            _finite(self.length_scale, "length_scale", minimum=np.finfo(float).tiny)
        if self.noise_scale is not None:
            _finite(self.noise_scale, "noise_scale", minimum=0.0)
        if self.noise_w_scale is not None:
            _finite(self.noise_w_scale, "noise_w_scale", minimum=0.0)
        if self.noise_w is not None:
            if self.noise_w_scale is not None:
                raise InvalidSynthesisConfigError("noise_w and noise_w_scale cannot both be set")
            _finite(self.noise_w, "noise_w", minimum=0.0)
            warnings.warn(
                "noise_w is deprecated; use noise_w_scale instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if not isinstance(self.normalize_audio, bool):
            raise InvalidSynthesisConfigError("normalize_audio must be a bool")
        _finite(self.volume, "volume", minimum=0.0)

    @property
    def resolved_noise_w_scale(self) -> float | None:
        return self.noise_w_scale if self.noise_w is None else self.noise_w


@dataclass(slots=True, init=False)
class AudioChunk:
    """One synthesized sentence or audio unit."""

    sample_rate: int
    audio_float_array: np.ndarray
    phonemes: tuple[str, ...] = ()
    phoneme_ids: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        sample_rate: int,
        audio_float_array: np.ndarray | None = None,
        phonemes: tuple[str, ...] = (),
        phoneme_ids: tuple[int, ...] = (),
        warnings: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        *,
        audio: np.ndarray | None = None,
    ) -> None:
        if audio_float_array is not None and audio is not None:
            raise ValueError("audio and audio_float_array cannot both be set")
        value = audio_float_array if audio_float_array is not None else audio
        if value is None:
            raise TypeError("audio_float_array is required")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 1:
            raise ModelInferenceError(f"audio must be one-dimensional, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ModelInferenceError("audio contains non-finite samples")
        self.sample_rate = sample_rate
        self.audio_float_array = array
        self.phonemes = tuple(phonemes)
        self.phoneme_ids = tuple(phoneme_ids)
        self.warnings = tuple(warnings)
        self.metadata = dict(metadata or {})

    @property
    def audio(self) -> np.ndarray:
        return self.audio_float_array

    @audio.setter
    def audio(self, value: np.ndarray) -> None:
        self.audio_float_array = np.asarray(value, dtype=np.float32)

    @property
    def sample_width(self) -> int:
        return 2

    @property
    def sample_channels(self) -> int:
        return 1

    @property
    def audio_int16_array(self) -> np.ndarray:
        return float_to_int16(self.audio_float_array)

    @property
    def audio_int16_bytes(self) -> bytes:
        return audio_to_int16_bytes(self.audio_float_array)

    @property
    def duration_seconds(self) -> float:
        return self.audio_float_array.size / self.sample_rate

    @property
    def int16(self) -> np.ndarray:
        return self.audio_int16_array

    @property
    def int16_bytes(self) -> bytes:
        return self.audio_int16_bytes


@dataclass(slots=True)
class AudioResult:
    """Owned final audio and provenance for one pipeline run."""

    audio: np.ndarray
    sample_rate: int
    source_text: str
    prepared_text: str
    chunks: list[AudioChunk] = field(default_factory=list)
    warnings: tuple[str, ...] = ()
    diagnostics: RuntimeDiagnostics | None = None
    timing: TimingDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("result audio must be one-dimensional and finite")
        self.chunks = list(self.chunks)
        self.warnings = tuple(self.warnings)
        self.metadata = dict(self.metadata)

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

    def play(self, *, wait: bool = True) -> None:
        """Play this result using the optional sounddevice dependency."""

        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Audio playback requires sounddevice. Install pipersynth[playback]."
            ) from exc
        sd.play(self.audio, self.sample_rate, blocking=wait)

    def release_audio(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
        self.chunks.clear()


@dataclass(frozen=True, slots=True)
class AudioUnitDescriptor:
    """Stable metadata describing a streamable audio unit."""

    index: int
    unit_kind: Literal["sentence", "paragraph"]
    text: str
    char_start: int | None = None
    char_end: int | None = None


@dataclass(slots=True)
class AudioUnitResult:
    """Audio and metadata for one streamable unit."""

    descriptor: AudioUnitDescriptor
    audio: np.ndarray
    sample_rate: int
    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1 or not np.all(np.isfinite(self.audio)):
            raise ModelInferenceError("unit audio must be one-dimensional and finite")
        self.phonemes = tuple(self.phonemes)
        self.phoneme_ids = tuple(self.phoneme_ids)
        self.warnings = tuple(self.warnings)
        self.metadata = dict(self.metadata)

    @property
    def audio_int16_array(self) -> np.ndarray:
        return float_to_int16(self.audio)

    @property
    def audio_int16_bytes(self) -> bytes:
        return audio_to_int16_bytes(self.audio)

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate
    def play(self, *, wait: bool = True) -> None:
        """Play this unit using the optional sounddevice dependency."""

        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Audio playback requires sounddevice. Install pipersynth[playback]."
            ) from exc
        sd.play(self.audio, self.sample_rate, blocking=wait)


    def release_audio(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
