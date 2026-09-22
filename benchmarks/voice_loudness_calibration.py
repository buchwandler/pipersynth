#!/usr/bin/env python3
"""Promote a reviewed Piper voice loudness report into runtime data."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipersynth.voice_level import VoiceCalibrationKey

PRODUCTION_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "pipersynth" / "data" / "voice_level_calibration.json"
)
REVIEWED_STATUSES = frozenset({"eligible"})
_RUNTIME_FIELDS = {
    "gain_db",
    "measured_lufs",
    "reference_lufs",
    "mad_lu",
    "samples",
    "method",
    "corpus_version",
}


class PromotionError(ValueError):
    """Raised when a report is not safe to promote."""


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_measurement_report(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as exc:
        raise PromotionError("measurement report is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PromotionError("measurement report must be an object")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise PromotionError(f"{name} must be a finite number")
    return converted


def _key(row: Mapping[str, Any]) -> VoiceCalibrationKey:
    values = [row.get(field) for field in ("model_source", "model_id", "quality", "speaker_id")]
    if not all(
        isinstance(value, (str, int)) and not isinstance(value, bool) for value in values[:3]
    ):
        raise PromotionError("aggregate rows must have non-empty voice identity fields")
    speaker_id = values[3]
    if not isinstance(speaker_id, int) or speaker_id < 0:
        raise PromotionError("aggregate rows require a non-negative numeric speaker_id")
    model_source, model_id, quality = values[:3]
    if not all(isinstance(value, str) and value for value in (model_source, model_id, quality)):
        raise PromotionError("aggregate identity strings must be non-empty")
    return VoiceCalibrationKey(model_source, model_id, quality, f"speaker-{speaker_id}")


def _validate_policy(policy: Mapping[str, Any]) -> None:
    required = (
        "repeats",
        "reference_lufs",
        "calibration_peak_ceiling_dbtp",
        "max_boost_db",
        "max_attenuation_db",
        "max_mad_lu",
    )
    for field in required:
        if field not in policy:
            raise PromotionError(f"policy requires {field}")
    repeats = policy["repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise PromotionError("policy.repeats must be a positive integer")
    for field in required[1:]:
        _finite(policy[field], f"policy.{field}")


def validate_measurement_report(
    report: Mapping[str, Any], *, require_complete_coverage: bool = True
) -> None:
    if report.get("schema") != 2:
        raise PromotionError("measurement report must use schema 2")
    if not isinstance(report.get("corpus"), str) or not report["corpus"]:
        raise PromotionError("measurement report requires a non-empty corpus")
    generated_with = report.get("generated_with")
    if not isinstance(generated_with, Mapping) or not generated_with:
        raise PromotionError("measurement report requires generated_with provenance")
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in generated_with.items()
    ):
        raise PromotionError("generated_with provenance must be non-empty string mappings")
    policy = report.get("policy")
    if not isinstance(policy, Mapping):
        raise PromotionError("measurement report requires a policy object")
    _validate_policy(policy)
    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping):
        raise PromotionError("measurement report requires coverage")
    expected = coverage.get("catalog_identities_expected")
    measured = coverage.get("catalog_identities_measured")
    failed = coverage.get("catalog_identities_failed")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (expected, measured, failed)
    ):
        raise PromotionError("coverage identity counts must be non-negative integers")
    if measured > expected or expected - measured > failed:
        raise PromotionError("coverage identity counts are inconsistent")
    complete = coverage.get("complete")
    if complete is not True and complete is not False:
        raise PromotionError("coverage.complete must be a boolean")
    if require_complete_coverage:
        if expected != measured:
            raise PromotionError("catalog identity coverage is incomplete")
        if failed != 0 or complete is not True:
            raise PromotionError("catalog identity failures prevent promotion")
    elif complete is True and (expected != measured or failed != 0):
        raise PromotionError("complete coverage cannot contain missing or failed identities")
    aggregates = report.get("aggregates")
    if not isinstance(aggregates, list):
        raise PromotionError("measurement report requires aggregates")
    seen: set[VoiceCalibrationKey] = set()
    for index, aggregate in enumerate(aggregates):
        if not isinstance(aggregate, Mapping):
            raise PromotionError(f"aggregate {index} must be an object")
        key = _key(aggregate)
        if key in seen:
            raise PromotionError(f"duplicate aggregate voice key: {key}")
        seen.add(key)
        status = aggregate.get("status")
        if not isinstance(status, str) or not status:
            raise PromotionError(f"aggregate {key} requires a status")
        repeats = aggregate.get("repeat_count")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
            raise PromotionError(f"aggregate {key} has invalid repeat_count")
        for field in ("median_lufs", "mad_lu", "max_true_peak_dbtp"):
            _finite(aggregate.get(field), f"aggregate {key}.{field}")


def _runtime_record(
    aggregate: Mapping[str, Any], report: Mapping[str, Any], gain_db: float
) -> dict[str, Any]:
    policy = report["policy"]
    return {
        "gain_db": gain_db,
        "measured_lufs": _finite(aggregate["median_lufs"], "median_lufs"),
        "reference_lufs": _finite(policy["reference_lufs"], "reference_lufs"),
        "mad_lu": _finite(aggregate["mad_lu"], "mad_lu"),
        "samples": aggregate["repeat_count"],
        "method": "bs1770",
        "corpus_version": report["corpus"],
    }


def build_runtime_calibration(
    report: Mapping[str, Any],
    *,
    reviewed_statuses: frozenset[str] = REVIEWED_STATUSES,
    require_complete_coverage: bool = True,
) -> dict[str, Any]:
    validate_measurement_report(report, require_complete_coverage=require_complete_coverage)
    policy = report["policy"]
    repeats = policy["repeats"]
    reference = _finite(policy["reference_lufs"], "policy.reference_lufs")
    ceiling = _finite(
        policy["calibration_peak_ceiling_dbtp"], "policy.calibration_peak_ceiling_dbtp"
    )
    voices: dict[str, dict[str, Any]] = {}
    for aggregate in report["aggregates"]:
        if aggregate["status"] not in reviewed_statuses or aggregate["repeat_count"] != repeats:
            continue
        measured = _finite(aggregate["median_lufs"], "aggregate.median_lufs")
        true_peak = _finite(aggregate["max_true_peak_dbtp"], "aggregate.max_true_peak_dbtp")
        requested = reference - measured
        gain = min(requested, ceiling - true_peak) if requested > 0 else requested
        key = str(_key(aggregate))
        if key in voices:
            raise PromotionError(f"duplicate runtime calibration key: {key}")
        voices[key] = _runtime_record(aggregate, report, gain)
    return {
        "schema": 1,
        "method": "bs1770",
        "corpus": report["corpus"],
        "reference_lufs": reference,
        "generated_with": dict(sorted(report["generated_with"].items())),
        "voices": {key: voices[key] for key in sorted(voices)},
    }


def write_runtime_calibration(report_path: Path | str, output_path: Path | str) -> None:
    output = Path(output_path)
    if output.resolve() == PRODUCTION_CALIBRATION_PATH.resolve():
        raise PromotionError("refusing to overwrite packaged production calibration data")
    catalog = build_runtime_calibration(load_measurement_report(report_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--review-status", action="append", default=[])
    args = parser.parse_args(argv)
    statuses = frozenset(args.review_status) if args.review_status else REVIEWED_STATUSES
    report = load_measurement_report(args.report)
    catalog = build_runtime_calibration(
        report,
        reviewed_statuses=statuses,
        require_complete_coverage=not args.allow_partial,
    )
    if args.output.resolve() == PRODUCTION_CALIBRATION_PATH.resolve():
        raise SystemExit("refusing to overwrite packaged production calibration data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
