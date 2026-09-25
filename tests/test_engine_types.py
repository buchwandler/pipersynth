from __future__ import annotations

import wave
from dataclasses import FrozenInstanceError
from io import BytesIO

import numpy as np
import pytest

from pipersynth.errors import InvalidSynthesisConfigError, ModelInferenceError
from pipersynth.types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedChunk,
    RenderedSegment,
    SynthesisConfig,
    SynthesisSegment,
)
from pipersynth.voice_level import VoiceLevelConfig


def test_synthesis_segment_preserves_request_coordinates_and_is_immutable() -> None:
    segment = SynthesisSegment(
        id="line-1",
        text="hello hello",
        language="en-us",
        speaker="speaker_2",
        pronunciation_overrides=(PronunciationOverride(6, 11, phonemes="həˈloʊ"),),
        annotations=(LinguisticToken(0, 5, text="hello", morph="Number=Sing"),),
    )
    assert segment.id == "line-1"
    assert (
        segment.text[
            segment.pronunciation_overrides[0].start : segment.pronunciation_overrides[0].end
        ]
        == "hello"
    )
    assert segment.annotations[0].morph == "Number=Sing"
    with pytest.raises(FrozenInstanceError):
        segment.id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs", [{"id": "", "text": "x", "language": "en"}, {"id": "x", "text": "x", "language": ""}]
)
def test_synthesis_segment_requires_identity_and_language(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        SynthesisSegment(**kwargs)


def test_synthesis_segment_rejects_invalid_offsets_and_annotation_text() -> None:
    with pytest.raises(ValueError, match="span exceeds"):
        SynthesisSegment(
            "x", "hello", "en", pronunciation_overrides=(PronunciationOverride(0, 6, phonemes="h"),)
        )
    with pytest.raises(ValueError, match="does not match"):
        SynthesisSegment("x", "hello", "en", annotations=(LinguisticToken(0, 2, text="heh"),))


def test_pronunciation_override_requires_an_effect() -> None:
    with pytest.raises(ValueError, match="at least one effect"):
        PronunciationOverride(0, 1)
    with pytest.raises(ValueError, match="0 <= start < end"):
        LinguisticToken(2, 2)


def test_synthesis_config_contains_only_engine_controls() -> None:
    config = SynthesisConfig(
        length_scale=1.1,
        noise_scale=0.5,
        noise_w_scale=0.7,
        output_gain=0.9,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=1.5),
    )
    assert config.output_gain == 0.9
    assert config.voice_level.mode == "calibrated"
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(noise_w_scale=-0.1)
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(output_gain=float("nan"))
    with pytest.raises(TypeError):
        SynthesisConfig(noise_w=0.5)  # type: ignore[call-arg]


def test_voice_level_config_rejects_nonfinite_gain() -> None:
    with pytest.raises(InvalidSynthesisConfigError):
        VoiceLevelConfig(gain_db=float("inf"))


def test_rendered_result_validates_audio_and_writes_standalone_wav() -> None:
    original = np.array([0.25, -0.25], dtype=np.float64)
    chunk = RenderedChunk(0, original, 22050, "line-1", phonemes=("h",), phoneme_ids=(1,))
    result = RenderedSegment(
        id="line-1",
        audio=original,
        sample_rate=22050,
        text="hello",
        language="en-us",
        speaker_id=1,
        phonemes=("h",),
        phoneme_ids=(1,),
        chunks=(chunk,),
    )
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert result.duration_seconds == pytest.approx(2 / 22050)
    assert result.chunks[0].audio.dtype == np.float32

    target = BytesIO()
    result.save_wav(target)
    target.seek(0)
    with wave.open(target, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 2


def test_rendered_result_rejects_non_mono_or_nonfinite_audio() -> None:
    args = dict(
        id="line-1",
        sample_rate=22050,
        text="hello",
        language="en-us",
        speaker_id=None,
        phonemes=(),
        phoneme_ids=(),
    )
    with pytest.raises(ModelInferenceError, match="mono"):
        RenderedSegment(audio=np.zeros((2, 2)), **args)
    with pytest.raises(ModelInferenceError, match="non-finite"):
        RenderedSegment(audio=np.array([np.nan]), **args)


def test_rendered_chunk_requires_positive_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        RenderedChunk(0, np.zeros(1), 0, "line-1")
