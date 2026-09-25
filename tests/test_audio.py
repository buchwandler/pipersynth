from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from pipersynth.audio import (
    audio_to_int16_bytes,
    finish_audio,
    float_to_int16,
    prepare_audio,
    write_wav,
)
from pipersynth.errors import ModelInferenceError


def test_prepare_audio_validates_and_optionally_normalizes_peak() -> None:
    original = np.array([2.0, -1.0], dtype=np.float32)
    normalized = prepare_audio(original, normalize=True)
    np.testing.assert_allclose(normalized, [1.0, -0.5])
    np.testing.assert_array_equal(original, [2.0, -1.0])
    np.testing.assert_array_equal(prepare_audio(original, normalize=False), original)


def test_finish_audio_applies_explicit_gain_and_clips() -> None:
    audio = np.array([0.5, -0.5], dtype=np.float32)
    np.testing.assert_allclose(finish_audio(audio, output_gain=0.5), [0.25, -0.25])
    np.testing.assert_array_equal(finish_audio(audio, output_gain=4.0), [1.0, -1.0])
    with pytest.raises(ModelInferenceError, match="output_gain"):
        finish_audio(audio, output_gain=float("nan"))


def test_pcm_conversion_is_deterministic() -> None:
    audio = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(float_to_int16(audio), [-32767, 0, 32767])
    assert audio_to_int16_bytes(audio) == float_to_int16(audio).tobytes()


def test_wav_writer_supports_paths_and_binary_streams(tmp_path: Path) -> None:
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    path = tmp_path / "nested" / "speech.wav"
    path.parent.mkdir()
    write_wav(path, audio, 22050)
    stream = BytesIO()
    write_wav(stream, audio, 22050)
    for target in (str(path), stream):
        if hasattr(target, "seek"):
            target.seek(0)
        with wave.open(target, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 22050
            assert wav.getnframes() == 3
