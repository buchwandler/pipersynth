#!/usr/bin/env python3
"""Measure explicitly prepared speech for static Piper voice calibration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from audiosig import measure_loudness

from pipersynth import (
    PiperVoice,
    SynthesisConfig,
    VoiceAssetManager,
    VoiceLevelConfig,
)
from pipersynth.assets import VoiceMetadata
from pipersynth.voice_level import VoiceCalibrationKey

POLICY_PATH = Path(__file__).with_name("data") / "voice_level_policy.json"
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/measurements.json")


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != 1 or not isinstance(policy.get("name"), str):
        raise ValueError("voice-level policy has an unsupported schema")
    repeats = policy.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("policy repeats must be a positive integer")
    for name in ("reference_lufs", "min_gain_db", "max_gain_db", "max_mad_lu"):
        value = policy.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"policy {name} must be finite")
    if policy["min_gain_db"] > policy["max_gain_db"] or policy["max_mad_lu"] < 0:
        raise ValueError("voice-level policy limits are invalid")
    return policy


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def generated_with() -> dict[str, str]:
    return {
        name: _package_version(name) for name in ("pipersynth", "audiosig", "onnxvoice", "piperg2p")
    }


def expand_identities(voices: Sequence[VoiceMetadata]) -> list[dict[str, Any]]:
    entries = []
    for metadata in sorted(voices, key=lambda item: (item.language_code, item.id, item.quality)):
        names = {int(identifier): str(name) for name, identifier in metadata.speaker_id_map.items()}
        for speaker_id in range(max(1, metadata.num_speakers)):
            key = VoiceCalibrationKey(
                "piper", metadata.id, metadata.quality, f"speaker-{speaker_id}"
            )
            entries.append(
                {
                    "model_source": "piper",
                    "model_id": metadata.id,
                    "quality": metadata.quality,
                    "locale": metadata.language_code,
                    "speaker_id": speaker_id,
                    "speaker_name": names.get(speaker_id),
                    "calibration_key": str(key),
                }
            )
    return entries


def _failure(
    entry: Mapping[str, Any], repeat: int, error: Exception, *, phase: str
) -> dict[str, Any]:
    return {
        **dict(entry),
        "repeat": repeat,
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def measure_identities(
    entries: Sequence[Mapping[str, Any]],
    *,
    text: str,
    language: str,
    cache_dir: Path | None,
    offline: bool,
    refresh_catalog: bool,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["model_id"])].append(entry)

    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repeats = int(policy["repeats"])
    config = SynthesisConfig(voice_level=VoiceLevelConfig(mode="off"))
    completed = 0
    for model_id in sorted(grouped):
        model_entries = grouped[model_id]
        try:
            voice = PiperVoice.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                offline=offline,
                refresh_catalog=refresh_catalog,
            )
        except Exception as error:
            for entry in model_entries:
                for repeat in range(repeats):
                    failures.append(_failure(entry, repeat, error, phase="voice_open"))
            continue

        with voice:
            for entry in model_entries:
                for repeat in range(repeats):
                    completed += 1
                    print(
                        f"[{completed}/{len(entries) * repeats}] "
                        f"{entry['calibration_key']} repeat {repeat + 1}",
                        flush=True,
                    )
                    try:
                        result = voice.synthesize_text(
                            text,
                            language=language,
                            id=f"calibration-{entry['speaker_id']}-{repeat}",
                            speaker=int(entry["speaker_id"]),
                            config=config,
                        )
                        loudness = measure_loudness(result.audio, sample_rate=result.sample_rate)
                        integrated_lufs = float(loudness.integrated_lufs)
                        if not math.isfinite(integrated_lufs):
                            raise ValueError("integrated loudness is not finite")
                        measurements.append(
                            {
                                **dict(entry),
                                "repeat": repeat,
                                "sample_rate": result.sample_rate,
                                "duration_seconds": result.duration_seconds,
                                "integrated_lufs": integrated_lufs,
                            }
                        )
                    except Exception as error:
                        failures.append(_failure(entry, repeat, error, phase="synthesis"))
    return measurements, failures


def aggregate_measurements(
    measurements: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        groups[str(measurement["calibration_key"])].append(measurement)

    aggregates = []
    for _key, rows in sorted(groups.items()):
        values = [float(row["integrated_lufs"]) for row in rows]
        median_lufs = statistics.median(values)
        mad_lu = statistics.median(abs(value - median_lufs) for value in values)
        repeat_count = len(values)
        if repeat_count < int(policy["repeats"]):
            status = "incomplete"
        elif mad_lu > float(policy["max_mad_lu"]):
            status = "high_variability"
        else:
            status = "eligible"
        requested_gain = float(policy["reference_lufs"]) - median_lufs
        gain_db = min(
            float(policy["max_gain_db"]),
            max(float(policy["min_gain_db"]), requested_gain),
        )
        identity = rows[0]
        aggregates.append(
            {
                **{
                    field: identity[field]
                    for field in (
                        "model_source",
                        "model_id",
                        "quality",
                        "locale",
                        "speaker_id",
                        "speaker_name",
                        "calibration_key",
                    )
                },
                "median_lufs": median_lufs,
                "mad_lu": mad_lu,
                "repeat_count": repeat_count,
                "status": status,
                "gain_db": gain_db,
            }
        )
    return aggregates


def build_report(
    entries: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    text: str,
    language: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    aggregates = aggregate_measurements(measurements, policy)
    full_keys = {
        str(row["calibration_key"])
        for row in aggregates
        if row["repeat_count"] == policy["repeats"]
    }
    expected = len(entries)
    measured = len(full_keys)
    failed = len({str(row["calibration_key"]) for row in entries} - full_keys)
    return {
        "schema": 1,
        "corpus": str(policy["name"]),
        "stimulus": {"text": text, "language": language, "prepared": True},
        "generated_with": generated_with(),
        "policy": dict(policy),
        "coverage": {
            "catalog_identities_expected": expected,
            "catalog_identities_measured": measured,
            "catalog_identities_failed": failed,
            "failure_count": len(failures),
            "complete": expected == measured and failed == 0,
        },
        "measurements": list(measurements),
        "failures": list(failures),
        "aggregates": aggregates,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", help="PiperG2P language for the prepared stimulus")
    parser.add_argument("--text", help="already-prepared speakable stimulus text")
    parser.add_argument("--voice", help="one exact Piper catalog voice ID")
    parser.add_argument("--quality")
    parser.add_argument("--speaker", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--max-voices", type=int)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _normalized_language(value: str) -> str:
    return value.casefold().replace("_", "-")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy()
    manager = VoiceAssetManager(args.cache_dir, offline=args.offline)
    voices = manager.list_voices(quality=args.quality, refresh=args.refresh_catalog)
    if args.voice:
        voices = tuple(
            voice for voice in voices if voice.id == args.voice or args.voice in voice.aliases
        )
    if args.language:
        requested = _normalized_language(args.language)
        voices = tuple(
            voice for voice in voices if _normalized_language(voice.language_code) == requested
        )
    if args.max_voices is not None:
        voices = voices[: args.max_voices]
    entries = expand_identities(voices)
    if args.speaker is not None:
        entries = [entry for entry in entries if entry["speaker_id"] == args.speaker]
    if not entries:
        raise SystemExit("no catalog speaker identities matched the requested filters")
    if args.list_only:
        for entry in entries:
            print(entry["calibration_key"])
        return 0
    if not args.language or not args.text:
        raise SystemExit("--language and --text are required unless --list-only is used")

    measurements, failures = measure_identities(
        entries,
        text=args.text,
        language=args.language,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh_catalog=args.refresh_catalog,
        policy=policy,
    )
    report = build_report(
        entries,
        measurements,
        failures,
        text=args.text,
        language=args.language,
        policy=policy,
    )
    _write_report(args.output, report)
    print(f"Measurements: {len(measurements)}")
    print(f"Failures: {len(failures)}")
    print(f"Report: {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
