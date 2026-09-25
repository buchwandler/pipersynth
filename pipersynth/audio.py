from __future__ import annotations

import wave
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .errors import ModelInferenceError

AUDIO_PEAK_EPSILON = 1e-8


def _as_float32_1d(audio: np.ndarray, *, name: str = "audio") -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32)
    if result.ndim != 1:
        raise ModelInferenceError(f"{name} must be one-dimensional, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ModelInferenceError(f"{name} contains non-finite samples")
    return result


def prepare_audio(audio: np.ndarray, *, normalize: bool) -> np.ndarray:
    """Validate inference audio and optionally normalize its peak."""
    result = _as_float32_1d(audio)
    if normalize and result.size:
        peak = float(np.max(np.abs(result)))
        if peak >= AUDIO_PEAK_EPSILON:
            result = result / peak
        else:
            result = np.zeros_like(result)
    return result


def finish_audio(audio: np.ndarray, *, output_gain: float) -> np.ndarray:
    """Apply explicit request-local output gain and the safety clamp."""
    result = _as_float32_1d(audio)
    if (
        isinstance(output_gain, bool)
        or not isinstance(output_gain, (int, float))
        or not np.isfinite(output_gain)
        or output_gain < 0
    ):
        raise ModelInferenceError("output_gain must be a finite non-negative number")
    if output_gain != 1.0:
        result = result * np.float32(output_gain)
    if not np.all(np.isfinite(result)):
        raise ModelInferenceError("postprocessed audio contains non-finite samples")
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert finite normalized audio to clipped signed 16-bit PCM."""

    result = _as_float32_1d(audio)
    return (np.clip(result, -1.0, 1.0) * 32767.0).astype(np.int16)


def audio_to_int16_bytes(audio: np.ndarray) -> bytes:
    return float_to_int16(audio).tobytes()


def write_wav(
    target: str | Path | BinaryIO,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """Write mono 16-bit PCM WAV data to a path or binary file-like object."""

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    pcm = float_to_int16(audio)
    target_file = str(target) if isinstance(target, Path) else target
    with wave.open(target_file, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
