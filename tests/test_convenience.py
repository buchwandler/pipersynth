from __future__ import annotations

from pathlib import Path

import numpy as np

import pipersynth.convenience as convenience
from pipersynth.types import AudioResult


class FakePipeline:
    last = None
    instances = []

    def __init__(self):
        self.closed = False
        self.__class__.instances.append(self)

    @classmethod
    def from_pretrained(cls, voice, **kwargs):
        cls.last = (voice, kwargs)
        return cls()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def run(self, text):
        return AudioResult(np.asarray([0.25, -0.25]), 22050, text, text)


def test_synthesize_to_wav_forwards_options_and_closes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(convenience, "PiperPipeline", FakePipeline)
    output = convenience.synthesize_to_wav(
        "hello",
        tmp_path / "nested" / "hello.wav",
        voice="test",
        speaker="alice",
        length_scale=0.9,
        providers=["CPUExecutionProvider"],
    )
    assert output.exists()
    assert FakePipeline.last[0] == "test"
    generation = FakePipeline.last[1]["generation"]
    assert generation.speaker == "alice"
    assert generation.length_scale == 0.9
    assert FakePipeline.instances[-1].closed


def test_failed_save_does_not_leave_temporary_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(convenience, "PiperPipeline", FakePipeline)
    monkeypatch.setattr(
        convenience, "write_wav", lambda *args: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    output = tmp_path / "failure.wav"
    try:
        convenience.synthesize_to_wav("hello", output, voice="test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected write failure")
    assert not output.exists()
    assert not tuple(tmp_path.glob("*.tmp"))
    assert FakePipeline.instances[-1].closed
