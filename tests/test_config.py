from __future__ import annotations

import numpy as np
import pytest

from pipersynth import SynthesisConfig, VoiceLevelConfig
from pipersynth.errors import InvalidSynthesisConfigError


def test_synthesis_config_accepts_engine_controls() -> None:
    config = SynthesisConfig(
        length_scale=1.0,
        noise_scale=0.5,
        noise_w_scale=0.8,
        normalize_audio=False,
        output_gain=1.25,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=-1.0),
    )
    assert config.length_scale == 1.0
    assert config.noise_scale == 0.5
    assert config.noise_w_scale == 0.8
    assert config.output_gain == 1.25
    assert config.voice_level.gain_db == -1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length_scale": 0.0},
        {"length_scale": np.inf},
        {"noise_scale": -1.0},
        {"noise_w_scale": np.nan},
        {"output_gain": -0.1},
        {"output_gain": True},
        {"normalize_audio": 1},
    ],
)
def test_synthesis_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(**kwargs)


def test_removed_aliases_and_document_controls_are_not_accepted() -> None:
    for name in ("noise_w", "speaker_id", "volume", "sentence_silence", "loudness"):
        with pytest.raises(TypeError):
            SynthesisConfig(**{name: 1.0})
