from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

import pipersynth
import pipersynth.convenience as convenience
from pipersynth import RenderedSegment, SynthesisConfig, VoiceLevelConfig


class FakePiperVoice:
    calls: list[tuple[str, dict[str, Any]]] = []
    request: tuple[str, dict[str, Any]] | None = None

    @classmethod
    def from_pretrained(cls, voice: str, **kwargs: Any) -> FakePiperVoice:
        cls.calls.append((voice, kwargs))
        return cls()

    def __enter__(self) -> FakePiperVoice:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def synthesize_text(self, prepared_text: str, **kwargs: Any) -> RenderedSegment:
        self.__class__.request = (prepared_text, kwargs)
        return RenderedSegment(
            id=kwargs.get("id") or "generated",
            audio=np.array([0.25, -0.25], dtype=np.float32),
            sample_rate=22050,
            text=prepared_text,
            language=kwargs["language"],
            speaker_id=1,
            phonemes=("h",),
            phoneme_ids=(1,),
        )


def test_public_api_exports_only_engine_request_surface() -> None:
    assert pipersynth.PiperVoice
    assert pipersynth.SynthesisSegment
    assert pipersynth.RenderedSegment
    for removed in (
        "PiperPipeline",
        "PipelineConfig",
        "GenerationConfig",
        "UtterancePlan",
        "UtterancePlanner",
        "AudioResult",
        "AudioUnitResult",
        "LoudnessConfig",
        "PreparedTextResult",
    ):
        assert not hasattr(pipersynth, removed)


def test_package_import_without_removed_dependencies() -> None:
    command = """\
import importlib.abc
import sys

blocked = {"utterplan", "audiocompose", "ssmd"}

class BlockRemovedDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockRemovedDependencies())
import pipersynth
assert not blocked.intersection(sys.modules)
assert not hasattr(pipersynth, "PiperPipeline")
"""
    result = subprocess.run(
        [sys.executable, "-c", command], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_convenience_synthesizes_prepared_text_through_piper_voice(monkeypatch) -> None:
    monkeypatch.setattr(convenience, "PiperVoice", FakePiperVoice)
    FakePiperVoice.calls.clear()
    result = convenience.synthesize(
        "Prepared text.",
        voice="en_US-lessac-medium",
        language="en-us",
        id="item-1",
        speaker="speaker_2",
        length_scale=0.9,
        output_gain=0.8,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=1.0),
        offline=True,
    )
    assert isinstance(result, RenderedSegment)
    assert result.text == "Prepared text."
    assert FakePiperVoice.calls[0][0] == "en_US-lessac-medium"
    assert FakePiperVoice.calls[0][1]["offline"] is True
    request = FakePiperVoice.request
    assert request is not None
    assert request[0] == "Prepared text."
    assert request[1]["language"] == "en-us"
    assert request[1]["id"] == "item-1"
    assert request[1]["speaker"] == "speaker_2"
    config = request[1]["config"]
    assert isinstance(config, SynthesisConfig)
    assert config.length_scale == 0.9
    assert config.output_gain == 0.8
    assert config.voice_level.gain_db == 1.0


def test_convenience_wav_helper_writes_independent_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(convenience, "PiperVoice", FakePiperVoice)
    output = convenience.synthesize_to_wav(
        "Prepared text.",
        tmp_path / "nested" / "speech.wav",
        voice="test-voice",
        language="en-us",
    )
    assert output.exists()
    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 2
    assert not tuple(output.parent.glob("*.tmp"))
