from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import audiosig
import numpy as np

from .loudness_config import LoudnessConfig

_SUPPORTED_SCHEMA = 1
_SUPPORTED_METHOD = "bs1770"
_RECORD_FIELDS = {
    "gain_db",
    "measured_lufs",
    "reference_lufs",
    "mad_lu",
    "samples",
    "method",
    "corpus_version",
}
_TOP_FIELDS = {"schema", "method", "corpus", "reference_lufs", "generated_with", "voices"}


class CalibrationDataError(ValueError):
    """Raised when a calibration catalog is malformed or unsafe."""


@dataclass(frozen=True, slots=True, order=True)
class VoiceCalibrationKey:
    model_source: str
    model_id: str
    quality: str
    voice: str

    def __post_init__(self) -> None:
        values = (self.model_source, self.model_id, self.quality, self.voice)
        if any(not isinstance(value, str) or not value for value in values):
            raise CalibrationDataError("calibration key components must be non-empty strings")

    def __str__(self) -> str:
        return ":".join((self.model_source, self.model_id, self.quality, self.voice))

    @classmethod
    def parse(cls, value: str) -> VoiceCalibrationKey:
        if not isinstance(value, str):
            raise CalibrationDataError("calibration key must be a string")
        parts = value.split(":")
        if len(parts) != 4:
            raise CalibrationDataError(
                "calibration key must have model_source:model_id:quality:voice form"
            )
        return cls(*parts)


@dataclass(frozen=True, slots=True)
class VoiceLevelCalibration:
    gain_db: float
    measured_lufs: float | None = None
    reference_lufs: float | None = None
    mad_lu: float | None = None
    samples: int | None = None
    method: str = _SUPPORTED_METHOD
    corpus_version: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceCalibrationCatalog:
    schema: int
    method: str
    corpus: str
    reference_lufs: float
    generated_with: Mapping[str, str]
    voices: Mapping[VoiceCalibrationKey, VoiceLevelCalibration]


VoiceLevelSource = Literal["off", "override", "catalog", "missing_identity", "missing_calibration"]


@dataclass(frozen=True, slots=True)
class VoiceLevelApplication:
    applied: bool
    gain_db: float
    source: VoiceLevelSource
    key: VoiceCalibrationKey | None


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationDataError(f"{name} must be a finite number")
    converted = float(value)
    if not np.isfinite(converted):
        raise CalibrationDataError(f"{name} must be finite")
    return converted


def _optional_finite(value: Any, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationDataError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_record(key: VoiceCalibrationKey, raw: Any) -> VoiceLevelCalibration:
    if not isinstance(raw, Mapping):
        raise CalibrationDataError(f"record for {key} must be an object")
    unknown = set(raw) - _RECORD_FIELDS
    if unknown:
        raise CalibrationDataError(f"record for {key} has unknown field(s): {sorted(unknown)}")
    if "gain_db" not in raw:
        raise CalibrationDataError(f"record for {key} is missing gain_db")
    method = raw.get("method", _SUPPORTED_METHOD)
    if method != _SUPPORTED_METHOD:
        raise CalibrationDataError(f"record for {key} has unsupported method: {method!r}")
    samples = raw.get("samples")
    if samples is not None:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise CalibrationDataError(f"record for {key} samples must be a positive integer")
    return VoiceLevelCalibration(
        gain_db=_finite(raw["gain_db"], f"{key}.gain_db"),
        measured_lufs=_optional_finite(raw.get("measured_lufs"), f"{key}.measured_lufs"),
        reference_lufs=_optional_finite(raw.get("reference_lufs"), f"{key}.reference_lufs"),
        mad_lu=_optional_finite(raw.get("mad_lu"), f"{key}.mad_lu"),
        samples=samples,
        method=method,
        corpus_version=raw.get("corpus_version"),
    )


def _validate_catalog(raw: Any) -> VoiceCalibrationCatalog:
    if not isinstance(raw, Mapping):
        raise CalibrationDataError("calibration catalog must be an object")
    unknown = set(raw) - _TOP_FIELDS
    if unknown:
        raise CalibrationDataError(f"calibration catalog has unknown field(s): {sorted(unknown)}")
    if raw.get("schema") != _SUPPORTED_SCHEMA:
        raise CalibrationDataError(f"unsupported calibration schema: {raw.get('schema')!r}")
    if raw.get("method") != _SUPPORTED_METHOD:
        raise CalibrationDataError(f"unsupported calibration method: {raw.get('method')!r}")
    corpus = raw.get("corpus")
    if not isinstance(corpus, str) or not corpus:
        raise CalibrationDataError("corpus must be a non-empty string")
    generated_with = raw.get("generated_with")
    if not isinstance(generated_with, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in generated_with.items()
    ):
        raise CalibrationDataError("generated_with must be a string mapping")
    voices_raw = raw.get("voices")
    if not isinstance(voices_raw, Mapping):
        raise CalibrationDataError("voices must be an object")
    voices: dict[VoiceCalibrationKey, VoiceLevelCalibration] = {}
    for raw_key, record in voices_raw.items():
        key = VoiceCalibrationKey.parse(raw_key)
        if key in voices:
            raise CalibrationDataError(f"duplicate normalized calibration key: {key}")
        voices[key] = _validate_record(key, record)
    return VoiceCalibrationCatalog(
        schema=_SUPPORTED_SCHEMA,
        method=_SUPPORTED_METHOD,
        corpus=corpus,
        reference_lufs=_finite(raw.get("reference_lufs"), "reference_lufs"),
        generated_with=dict(sorted(generated_with.items())),
        voices=dict(sorted(voices.items(), key=lambda item: str(item[0]))),
    )


def load_voice_calibration(path: Path | str) -> VoiceCalibrationCatalog:
    """Load and strictly validate a runtime calibration catalog."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as exc:
        raise CalibrationDataError(f"invalid calibration JSON: {source}") from exc
    return _validate_catalog(raw)


@lru_cache(maxsize=1)
def default_voice_calibration() -> VoiceCalibrationCatalog:
    resource = files("pipersynth").joinpath("data", "voice_level_calibration.json")
    try:
        with resource.open("r", encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as exc:
        raise CalibrationDataError(f"invalid calibration JSON: {resource}") from exc
    return _validate_catalog(raw)


def apply_voice_level_calibration(
    audio: np.ndarray,
    config: LoudnessConfig,
    key: VoiceCalibrationKey | None,
    *,
    catalog: VoiceCalibrationCatalog | None = None,
) -> tuple[np.ndarray, VoiceLevelApplication]:
    """Apply one deterministic static gain; never measure the waveform."""

    result = np.asarray(audio, dtype=np.float32)
    if config.voice_gain_db is not None:
        gain = float(config.voice_gain_db)
        source: VoiceLevelSource = "override"
    elif config.voice_leveling != "calibrated":
        gain = 0.0
        source = "off"
    elif key is None:
        gain = 0.0
        source = "missing_identity"
    else:
        selected = (catalog or default_voice_calibration()).voices.get(key)
        if selected is None:
            gain = 0.0
            source = "missing_calibration"
        else:
            gain = selected.gain_db
            source = "catalog"
    if gain:
        result = np.asarray(audiosig.apply_gain_db(result, gain, clip=False), dtype=np.float32)
    else:
        result = result.copy()
    return result, VoiceLevelApplication(bool(gain), gain, source, key)


__all__ = [
    "CalibrationDataError",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "load_voice_calibration",
]
