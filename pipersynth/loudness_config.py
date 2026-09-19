from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .errors import InvalidSynthesisConfigError

VoiceLevelingMode = Literal["off", "calibrated"]
PeakPolicy = Literal["reduce_gain", "error"]


def _validate_db(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSynthesisConfigError(f"{name} must be a finite number or None")
    if not np.isfinite(value):
        raise InvalidSynthesisConfigError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class LoudnessConfig:
    """Immutable voice-leveling and complete-output loudness policy."""

    voice_leveling: VoiceLevelingMode = "off"
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = -1.0
    peak_policy: PeakPolicy = "reduce_gain"
    voice_gain_db: float | None = None

    def __post_init__(self) -> None:
        if self.voice_leveling not in {"off", "calibrated"}:
            raise InvalidSynthesisConfigError("voice_leveling must be 'off' or 'calibrated'")
        if self.peak_policy not in {"reduce_gain", "error"}:
            raise InvalidSynthesisConfigError("peak_policy must be 'reduce_gain' or 'error'")
        _validate_db(self.target_lufs, "target_lufs")
        _validate_db(self.true_peak_ceiling_dbtp, "true_peak_ceiling_dbtp")
        _validate_db(self.voice_gain_db, "voice_gain_db")


def coerce_loudness(value: LoudnessConfig | Mapping[str, object] | None) -> LoudnessConfig:
    """Normalize a public loudness override into an immutable policy."""

    if value is None:
        return LoudnessConfig()
    if isinstance(value, LoudnessConfig):
        return value
    if isinstance(value, Mapping):
        return LoudnessConfig(**value)  # type: ignore[arg-type]
    raise TypeError("loudness must be a LoudnessConfig, mapping, or None")


__all__ = ["LoudnessConfig", "PeakPolicy", "VoiceLevelingMode", "coerce_loudness"]
