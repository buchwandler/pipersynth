from __future__ import annotations

import inspect
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pipersynth
import pipersynth.convenience as convenience
from pipersynth import (
    SynthesisConfig,
    SynthesisResult,
    TextChunkingConfig,
    VoiceLevelConfig,
)


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

    def synthesize_text(self, prepared_text: str, **kwargs: Any) -> SynthesisResult:
        self.__class__.request = (prepared_text, kwargs)
        return SynthesisResult(
            id=kwargs.get("id") or "generated",
            audio=np.array([0.25, -0.25], dtype=np.float32),
            sample_rate=22050,
            text=prepared_text,
            language=kwargs["language"],
            metadata={"speaker_id": 1, "phonemes": ("h",), "phoneme_ids": (1,)},
        )


def test_public_api_exports_only_engine_request_surface() -> None:
    assert pipersynth.PiperVoice
    assert pipersynth.SynthesisSegment
    assert pipersynth.SynthesisRequest
    assert pipersynth.SynthesisResult
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


def test_strict_synthesis_api_has_no_chunking_or_rendered_chunks() -> None:
    synthesize_parameters = inspect.signature(pipersynth.PiperVoice.synthesize).parameters
    text_parameters = inspect.signature(pipersynth.PiperVoice.synthesize_text).parameters
    assert "chunking" not in synthesize_parameters
    assert "chunking" not in text_parameters
    assert "request" in synthesize_parameters
    assert pipersynth.SynthesisResult.supports_timestamps is False
    assert not hasattr(pipersynth.SynthesisResult, "chunks")


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
    assert isinstance(result, SynthesisResult)
    assert result.text == "Prepared text."
    assert FakePiperVoice.calls[0][0] == "en_US-lessac-medium"
    assert FakePiperVoice.calls[0][1]["offline"] is True
    request = FakePiperVoice.request
    assert request is not None
    assert request[0] == "Prepared text."
    assert request[1]["language"] == "en-us"
    assert request[1]["id"] == "item-1"
    assert "chunking" not in request[1]
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
    assert FakePiperVoice.request is not None
    assert "chunking" not in FakePiperVoice.request[1]
    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 2
    assert not tuple(output.parent.glob("*.tmp"))


def test_public_text_chunking_config_validates_mode_and_limit() -> None:
    assert pipersynth.TextChunkingConfig is TextChunkingConfig
    assert TextChunkingConfig().mode == "none"
    assert TextChunkingConfig(mode="sentence", max_chars=120).max_chars == 120
    assert TextChunkingConfig(mode="none").max_chars is None

    with pytest.raises(ValueError, match="mode"):
        TextChunkingConfig(mode="all")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        TextChunkingConfig(max_chars=0)
    with pytest.raises(ValueError, match="positive integer"):
        TextChunkingConfig(max_chars=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires sentence"):
        TextChunkingConfig(mode="none", max_chars=10)


def test_strict_synthesis_does_not_import_phrasplit_in_subprocess() -> None:
    command = """\\
import builtins
from types import SimpleNamespace

original_import = builtins.__import__
def block_phrasplit(name, *args, **kwargs):
    if name.partition(".")[0] == "phrasplit":
        raise AssertionError("Phrasplit must not be imported in none mode")
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_phrasplit

import numpy as np
from piperg2p import PhonemeSentence, VoiceConfig
from pipersynth import PiperVoice

config = VoiceConfig.from_dict({
    "num_symbols": 4,
    "num_speakers": 1,
    "audio": {"sample_rate": 22050},
    "phoneme_type": "text",
    "espeak": {"voice": "en-us"},
    "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]},
})
class Runtime:
    def infer(self, ids, **kwargs):
        return SimpleNamespace(
            audio=np.array([0.1], dtype=np.float32), sample_rate=22050
        )
class G2P:
    def phonemize_prepared(self, text, **kwargs):
        return SimpleNamespace(
            sentences=(PhonemeSentence(("a",), (3,)),),
            warnings=(),
            diagnostics=None,
        )
voice = PiperVoice(
    Runtime(), config, g2p_factory=lambda language, *, config: G2P()
 )
result = voice.synthesize_text(
    "prepared text",
    language="en-us",
 )
assert result.audio.size == 1
assert result.word_timings == ()
voice.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
