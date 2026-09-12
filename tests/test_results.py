import numpy as np

from pipersynth import AudioResult, AudioUnitDescriptor, AudioUnitResult


def test_result_and_unit_release_audio() -> None:
    result = AudioResult(np.array([0.1, 0.2], dtype=np.float32), 10, "source", "prepared")
    unit = AudioUnitResult(
        AudioUnitDescriptor(0, "sentence", "source"),
        np.array([0.1], dtype=np.float32),
        10,
        (),
        (),
    )
    result.release_audio()
    unit.release_audio()
    assert result.audio.size == 0
    assert not result.chunks
    assert unit.audio.size == 0
