import wave
from io import BytesIO

import numpy as np
import pytest

from pipersynth import AudioChunk
from pipersynth.audio import float_to_int16, postprocess_audio, silence_samples, write_wav
from pipersynth.errors import ModelInferenceError


def test_pcm_conversion_clips_and_maps_normalized_values() -> None:
    assert float_to_int16(np.array([-1.0, 0.0, 1.0], dtype=np.float32)).tolist() == [
        -32767,
        0,
        32767,
    ]
    assert float_to_int16(np.array([-2.0, 2.0], dtype=np.float32)).tolist() == [-32767, 32767]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_pcm_conversion_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ModelInferenceError, match="non-finite"):
        float_to_int16(np.array([value], dtype=np.float32))


def test_postprocess_normalizes_then_applies_volume_and_clips() -> None:
    audio = postprocess_audio(np.array([-2.0, 0.5], dtype=np.float32), normalize=True, volume=2.0)
    assert audio.tolist() == [-1.0, 0.5]


def test_silence_sample_count_uses_rounding() -> None:
    assert silence_samples(22050, 0.1) == 2205


def test_wav_writes_mono_pcm_to_file_like_object() -> None:
    target = BytesIO()
    write_wav(target, np.array([-1.0, 0.0, 1.0], dtype=np.float32), 22050)
    target.seek(0)
    with wave.open(target, "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 22050
        assert handle.readframes(3) == np.array([-32767, 0, 32767], dtype="<i2").tobytes()


def test_audio_chunk_exposes_public_and_compatibility_properties() -> None:
    chunk = AudioChunk(22050, np.array([0.0, 0.5], dtype=np.float32))
    assert chunk.audio is chunk.audio_float_array
    assert chunk.int16.tolist() == [0, 16383]
    assert chunk.int16_bytes == chunk.audio_int16_bytes
    assert chunk.sample_width == 2
    assert chunk.sample_channels == 1
    assert chunk.duration_seconds == 2 / 22050
