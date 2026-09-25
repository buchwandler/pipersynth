#!/usr/bin/env python3
"""Build a candidate static calibration catalog from a measured report."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipersynth.voice_level import VoiceCalibrationKey

PRODUCTION_CATALOG = (
    Path(__file__).resolve().parents[1] / "pipersynth" / "data" / "voice_level_calibration.json"
)
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/candidate_catalog.json")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def build_candidate(
    report: Mapping[str, Any], *, allow_partial: bool, include_high_variability: bool
) -> tuple[dict[str, Any], dict[str, int]]:
    if report.get("schema") != 1:
        raise ValueError("unsupported measurement report schema")
    policy = report.get("policy")
    coverage = report.get("coverage")
    stimulus = report.get("stimulus")
    if not isinstance(policy, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("measurement report is missing policy or coverage")
    if not isinstance(stimulus, Mapping) or stimulus.get("prepared") is not True:
        raise ValueError("report must identify its stimulus as explicitly prepared")
    corpus = report.get("corpus")
    if not isinstance(corpus, str) or not corpus:
        raise ValueError("report corpus must be a non-empty string")
    reference_lufs = _finite_number(policy.get("reference_lufs"), "reference_lufs")
    min_gain_db = _finite_number(policy.get("min_gain_db"), "min_gain_db")
    max_gain_db = _finite_number(policy.get("max_gain_db"), "max_gain_db")
    max_mad_lu = _finite_number(policy.get("max_mad_lu"), "max_mad_lu")
    repeats = _non_negative_integer(policy.get("repeats"), "repeats", minimum=1)
    expected = _non_negative_integer(coverage.get("catalog_identities_expected"), "expected")
    measured = _non_negative_integer(coverage.get("catalog_identities_measured"), "measured")
    failed = _non_negative_integer(coverage.get("catalog_identities_failed"), "failed")
    if min_gain_db > max_gain_db or max_mad_lu < 0:
        raise ValueError("measurement policy limits are invalid")
    identity_overrides = policy.get("identity_overrides", {})
    if not isinstance(identity_overrides, Mapping):
        raise ValueError("measurement identity_overrides must be an object")
    for identity_key, identity_override in identity_overrides.items():
        if not isinstance(identity_key, str) or not isinstance(identity_override, Mapping):
            raise ValueError("measurement identity overrides must map keys to objects")
        try:
            VoiceCalibrationKey.parse(identity_key)
        except ValueError as error:
            raise ValueError(
                f"invalid measurement identity override key: {identity_key}"
            ) from error
        override_min_gain_db = _finite_number(
            identity_override.get("min_gain_db"), f"{identity_key}.min_gain_db"
        )
        if override_min_gain_db > max_gain_db:
            raise ValueError(f"{identity_key}.min_gain_db exceeds max_gain_db")
        _required_string(identity_override.get("rationale"), f"{identity_key}.rationale")
    if expected <= 0 or measured > expected or failed != expected - measured:
        raise ValueError("measurement coverage counts are inconsistent")
    if not coverage.get("complete") and not allow_partial:
        raise ValueError("incomplete identity coverage requires --allow-partial")
    if coverage.get("complete") != (measured == expected and failed == 0):
        raise ValueError("measurement coverage completeness flag is inconsistent")

    aggregates = report.get("aggregates")
    generated_with = report.get("generated_with")
    if not isinstance(aggregates, list) or not aggregates:
        raise ValueError("report contains no measured aggregates")
    if not isinstance(generated_with, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in generated_with.items()
    ):
        raise ValueError("report generated_with must be a string mapping")

    voices: dict[str, dict[str, Any]] = {}
    counts = {"eligible": 0, "high_variability": 0, "incomplete": 0}
    for aggregate in aggregates:
        if not isinstance(aggregate, Mapping):
            raise ValueError("aggregate entries must be objects")
        model_source = _required_string(aggregate.get("model_source"), "model_source")
        model_id = _required_string(aggregate.get("model_id"), "model_id")
        quality = _required_string(aggregate.get("quality"), "quality")
        speaker_id = _non_negative_integer(aggregate.get("speaker_id"), "speaker_id")
        key = VoiceCalibrationKey(model_source, model_id, quality, f"speaker-{speaker_id}")
        if aggregate.get("calibration_key") != str(key) or str(key) in voices:
            raise ValueError(f"invalid or duplicate calibration key: {key}")

        median_lufs = _finite_number(aggregate.get("median_lufs"), f"{key}.median_lufs")
        mad_lu = _finite_number(aggregate.get("mad_lu"), f"{key}.mad_lu")
        gain_db = _finite_number(aggregate.get("gain_db"), f"{key}.gain_db")
        repeat_count = _non_negative_integer(aggregate.get("repeat_count"), f"{key}.repeat_count")
        identity_override = identity_overrides.get(str(key))
        identity_min_gain_db = (
            float(identity_override["min_gain_db"])
            if identity_override is not None
            else min_gain_db
        )
        expected_gain = min(
            max_gain_db,
            max(identity_min_gain_db, reference_lufs - median_lufs),
        )
        if not math.isclose(gain_db, expected_gain, abs_tol=1e-9):
            raise ValueError(f"{key}.gain_db does not match the declared policy")

        status = aggregate.get("status")
        if status == "incomplete" or repeat_count < repeats:
            counts["incomplete"] += 1
            continue
        if status == "high_variability" or mad_lu > max_mad_lu:
            counts["high_variability"] += 1
            if not include_high_variability:
                continue
        elif status != "eligible":
            raise ValueError(f"{key} has unknown review status {status!r}")
        else:
            counts["eligible"] += 1

        voices[str(key)] = {
            "gain_db": gain_db,
            "measured_lufs": median_lufs,
            "reference_lufs": reference_lufs,
            "mad_lu": mad_lu,
            "samples": repeat_count,
            "method": "bs1770",
            "corpus_version": corpus,
        }

    if counts["incomplete"] and not allow_partial:
        raise ValueError("incomplete speaker measurements require --allow-partial")
    if not voices:
        raise ValueError("no eligible speaker measurements to promote")
    catalog = {
        "schema": 1,
        "method": "bs1770",
        "corpus": corpus,
        "reference_lufs": reference_lufs,
        "generated_with": dict(sorted(generated_with.items())),
        "voices": dict(sorted(voices.items())),
    }
    return catalog, counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--include-high-variability", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output == PRODUCTION_CATALOG.resolve():
        raise SystemExit("promotion never writes directly to the packaged production catalog")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise SystemExit("measurement report must contain a JSON object")
    candidate, counts = build_candidate(
        report,
        allow_partial=args.allow_partial,
        include_high_variability=args.include_high_variability,
    )
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Eligible: {counts['eligible']}")
    print(f"High variability: {counts['high_variability']}")
    print(f"Incomplete: {counts['incomplete']}")
    print(f"Candidate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
