import warnings

import numpy as np
import pytest

from pipersynth import SynthesisConfig
from pipersynth.errors import InvalidSynthesisConfigError


def test_synthesis_config_accepts_valid_values() -> None:
    config = SynthesisConfig(
        speaker_id=2,
        length_scale=1.0,
        noise_scale=0.5,
        noise_w_scale=0.8,
        volume=1.25,
    )
    assert config.resolved_noise_w_scale == 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speaker_id": True},
        {"length_scale": 0.0},
        {"length_scale": np.inf},
        {"noise_scale": -1.0},
        {"noise_w_scale": np.nan},
        {"volume": -0.1},
        {"volume": True},
        {"normalize_audio": 1},
    ],
)
def test_synthesis_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(**kwargs)


def test_noise_w_is_a_deprecated_compatibility_alias() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = SynthesisConfig(noise_w=0.7)
    assert config.resolved_noise_w_scale == 0.7
    assert any(item.category is DeprecationWarning for item in caught)


def test_noise_alias_conflict_is_rejected() -> None:
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(noise_w=0.7, noise_w_scale=0.8)
