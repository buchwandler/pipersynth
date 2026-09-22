#!/usr/bin/env python3
"""Measure Piper catalog voices for offline, identity-specific leveling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from audiosig import measure_loudness

from pipersynth import GenerationConfig, LoudnessConfig, PiperPipeline, VoiceAssetManager
from pipersynth.assets import VoiceMetadata
from pipersynth.voice_level import VoiceCalibrationKey

POLICY_PATH = Path(__file__).with_name("data") / "voice_loudness_policy.json"
FALLBACKS_PATH = Path(__file__).with_name("data") / "voice_loudness_count_fallbacks.json"
COUNT_SOURCE = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
COUNT_VALUES = tuple(range(1, 11))
DEFAULT_OUTPUT = Path("benchmarks/output/voice_loudness")


class StimulusResolutionError(ValueError):
    """Raised when a locale has no safe spoken count stimulus."""


@dataclass(frozen=True, slots=True)
class LoudnessStimulus:
    locale: str
    normalized_language: str
    source: str
    spoken_text: str
    generator: str
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class StimulusPreflight:
    stimuli: Mapping[str, LoudnessStimulus]
    unsupported: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class BenchmarkPolicy:
    schema: int
    name: str
    first: int
    last: int
    repeats: int
    reference_lufs: float
    calibration_peak_ceiling_dbtp: float
    max_boost_db: float
    max_attenuation_db: float
    max_mad_lu: float


def load_policy(path: Path = POLICY_PATH) -> BenchmarkPolicy:
    data = json.loads(path.read_text(encoding="utf-8"))
    stimulus = data.get("stimulus", {})
    policy = BenchmarkPolicy(
        schema=int(data.get("schema", 0)),
        name=str(data.get("name", "")),
        first=int(stimulus.get("first", 0)),
        last=int(stimulus.get("last", 0)),
        repeats=int(data.get("repeats", 0)),
        reference_lufs=float(data["reference_lufs"]),
        calibration_peak_ceiling_dbtp=float(data["calibration_peak_ceiling_dbtp"]),
        max_boost_db=float(data["max_boost_db"]),
        max_attenuation_db=float(data["max_attenuation_db"]),
        max_mad_lu=float(data["max_mad_lu"]),
    )
    if policy.schema != 2 or not policy.name or (policy.first, policy.last) != (1, 10):
        raise ValueError("voice loudness policy must define schema 2 count stimulus 1 through 10")
    if policy.repeats < 1 or policy.max_mad_lu < 0:
        raise ValueError("voice loudness policy has invalid repeats or MAD threshold")
    return policy


def load_fallbacks(path: Path = FALLBACKS_PATH) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("stimulus") != "count-1-to-10":
        raise ValueError("count fallback data has an invalid schema or stimulus")
    locales = data.get("locales")
    if not isinstance(locales, dict):
        raise ValueError("count fallback locales must be an object")
    return locales


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def generated_with() -> dict[str, str]:
    return {
        "pipersynth": _package_version("pipersynth"),
        "audiosig": _package_version("audiosig"),
        "onnxvoice": _package_version("onnxvoice"),
        "utterplan": _package_version("utterplan"),
        "spokenform": _package_version("spokenform"),
        "piperg2p": _package_version("piperg2p"),
    }


def _spoken_count_words(locale: str) -> tuple[str, tuple[str, ...]]:
    try:
        from spokenform import normalize_language, normalize_numbers
    except ImportError as exc:
        raise StimulusResolutionError("spokenform is unavailable") from exc
    language = normalize_language(locale)
    words = tuple(normalize_numbers(str(value), language=language) for value in COUNT_VALUES)
    if any(not word.strip() or any(character.isdigit() for character in word) for word in words):
        raise StimulusResolutionError(f"spokenform produced unresolved digits for {locale!r}")
    return language, words


def resolve_count_stimulus(
    locale: str, fallbacks: dict[str, dict[str, Any]] | None = None
) -> LoudnessStimulus:
    """Resolve spoken cardinal words, using only a reviewed locale fallback."""
    fallbacks = load_fallbacks() if fallbacks is None else fallbacks
    normalized = locale.replace("-", "_").lower()
    try:
        normalized_language, words = _spoken_count_words(locale)
        return LoudnessStimulus(
            locale=locale,
            normalized_language=normalized_language,
            source=COUNT_SOURCE,
            spoken_text=", ".join(words) + ".",
            generator="spokenform",
            fallback_used=False,
        )
    except Exception as error:
        base = normalized.split("_", 1)[0]
        fallback = fallbacks.get(locale) or fallbacks.get(normalized) or fallbacks.get(base)
        if not isinstance(fallback, Mapping) or not isinstance(fallback.get("text"), str):
            raise StimulusResolutionError(f"no reviewed count fallback for {locale!r}") from error
        text = fallback["text"]
        if any(character.isdigit() for character in text):
            raise StimulusResolutionError(f"fallback for {locale!r} contains digits") from None
        return LoudnessStimulus(
            locale=locale,
            normalized_language=normalized,
            source=COUNT_SOURCE,
            spoken_text=text,
            generator="fallback",
            fallback_used=True,
        )


def preflight_stimuli(
    entries: Sequence[Mapping[str, Any]],
    fallbacks: dict[str, dict[str, Any]] | None = None,
) -> StimulusPreflight:
    """Resolve every distinct locale before any synthesis starts."""
    fallbacks = load_fallbacks() if fallbacks is None else fallbacks
    stimuli: dict[str, LoudnessStimulus] = {}
    unsupported: dict[str, str] = {}
    for locale in sorted({str(entry["locale"]) for entry in entries}):
        try:
            stimuli[locale] = resolve_count_stimulus(locale, fallbacks)
        except StimulusResolutionError as error:
            unsupported[locale] = str(error)
    return StimulusPreflight(stimuli=stimuli, unsupported=unsupported)


def stimulus_failures_for_entries(
    entries: Sequence[Mapping[str, Any]],
    unsupported: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Represent every identity whose locale failed stimulus preflight."""
    return [
        {
            **dict(entry),
            "status": "unsupported_stimulus",
            "phase": "stimulus_preflight",
            "error_type": "StimulusResolutionError",
            "error": unsupported[str(entry["locale"])],
        }
        for entry in entries
        if str(entry["locale"]) in unsupported
    ]


def expand_speaker_ids(metadata: VoiceMetadata) -> tuple[int, ...]:
    return tuple(range(max(1, metadata.num_speakers)))


def inventory_entries(
    voices: Sequence[VoiceMetadata],
    *,
    voice: str | None = None,
    language: str | None = None,
    locale: str | None = None,
    quality: str | None = None,
    speaker: int | None = None,
    max_voices: int | None = None,
) -> list[dict[str, Any]]:
    """Expand catalog metadata into one exact identity per catalog speaker."""

    selected = list(voices)
    if voice is not None:
        selected = [item for item in selected if item.id == voice or voice in item.aliases]
    if language is not None:
        selected = [item for item in selected if item.language_code == language]
    if locale is not None:
        selected = [item for item in selected if item.language_code == locale]
    if quality is not None:
        selected = [item for item in selected if item.quality == quality]
    if max_voices is not None:
        selected = selected[:max_voices]
    entries: list[dict[str, Any]] = []
    for metadata in sorted(selected, key=lambda item: item.id):
        ids = expand_speaker_ids(metadata)
        if speaker is not None:
            ids = tuple(item for item in ids if item == speaker)
        names = {identifier: name for name, identifier in metadata.speaker_id_map.items()}
        for speaker_id in ids:
            key = VoiceCalibrationKey(
                "piper", metadata.id, metadata.quality, f"speaker-{speaker_id}"
            )
            entries.append(
                {
                    "model_source": "piper",
                    "model_id": metadata.id,
                    "quality": metadata.quality,
                    "speaker_id": speaker_id,
                    "speaker_name": names.get(speaker_id),
                    "calibration_key": str(key),
                    "locale": metadata.language_code,
                    "language": metadata.language_code,
                    "source_revision": metadata.source_revision,
                    "num_speakers": metadata.num_speakers,
                }
            )
    return entries


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) if audio.size else 0.0
    return -math.inf if rms == 0 else 20 * math.log10(rms)


def _failure_row(
    entry: Mapping[str, Any],
    repeat: int,
    *,
    phase: str,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    return {
        **dict(entry),
        "repeat": repeat,
        "status": "failed",
        "phase": phase,
        "error_type": error_type,
        "error": error,
    }


def measure_model(
    entries: Sequence[Mapping[str, Any]],
    *,
    stimulus: LoudnessStimulus,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    repeats: int = 3,
    refresh_catalog: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure every speaker identity for one Piper model."""
    if not entries:
        raise ValueError("model worker requires at least one identity")
    if repeats < 1:
        raise ValueError("model worker requires at least one repeat")
    model_ids = {str(entry["model_id"]) for entry in entries}
    locales = {str(entry["locale"]) for entry in entries}
    if len(model_ids) != 1:
        raise ValueError("model worker received multiple model IDs")
    if len(locales) != 1:
        raise ValueError("model worker received multiple locales")
    if stimulus.locale not in locales:
        raise ValueError("model worker stimulus locale does not match model locale")

    model_id = next(iter(model_ids))
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pipeline: PiperPipeline | None = None
    try:
        try:
            pipeline = PiperPipeline.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                offline=offline,
                refresh_catalog=refresh_catalog,
                generation=GenerationConfig(
                    speaker=int(entries[0]["speaker_id"]),
                    normalize_audio=True,
                    volume=1.0,
                ),
                loudness=LoudnessConfig(voice_leveling="off", target_lufs=None),
                language=str(entries[0]["locale"]),
                language_policy="allow",
            )
        except Exception as exc:
            for entry in entries:
                for repeat in range(repeats):
                    failures.append(
                        _failure_row(
                            entry,
                            repeat,
                            phase="pipeline_open",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    )
            return measurements, failures

        for entry in entries:
            for repeat in range(repeats):
                try:
                    result = pipeline.run(stimulus.spoken_text, speaker=int(entry["speaker_id"]))
                    audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    metrics = measure_loudness(audio, sample_rate=result.sample_rate)
                    metric_values = {
                        "integrated_lufs": float(metrics.integrated_lufs),
                        "sample_peak_dbfs": float(metrics.sample_peak_dbfs),
                        "true_peak_dbtp": float(metrics.true_peak_dbtp),
                        "rms_dbfs": _rms_dbfs(audio),
                    }
                    if not all(math.isfinite(value) for value in metric_values.values()):
                        failures.append(
                            _failure_row(
                                entry,
                                repeat,
                                phase="measurement",
                                error_type="NonFiniteMeasurement",
                                error=f"non-finite loudness metrics: {metric_values}",
                            )
                        )
                        continue
                    measurements.append(
                        {
                            **dict(entry),
                            "repeat": repeat,
                            "stimulus": stimulus.spoken_text,
                            "stimulus_generator": stimulus.generator,
                            "sample_rate": result.sample_rate,
                            **metric_values,
                            "duration_seconds": audio.size / result.sample_rate,
                            "pipersynth_version": _package_version("pipersynth"),
                            "audiosig_version": _package_version("audiosig"),
                            "onnxvoice_version": _package_version("onnxvoice"),
                            "utterplan_version": _package_version("utterplan"),
                        }
                    )
                except Exception as exc:
                    failures.append(
                        _failure_row(
                            entry,
                            repeat,
                            phase="synthesis",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    )
    finally:
        if pipeline is not None:
            pipeline.close()
    return measurements, failures


def measure_repeats(
    entry: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    repeats: int = 3,
    refresh_catalog: bool = False,
    stimulus: LoudnessStimulus,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper for measuring one identity."""
    return measure_model(
        [entry],
        cache_dir=cache_dir,
        offline=offline,
        repeats=repeats,
        refresh_catalog=refresh_catalog,
        stimulus=stimulus,
    )


def group_entries_by_model(
    entries: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group identities by model in deterministic model and speaker order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["model_id"]), []).append(dict(entry))
    return [
        (
            model_id,
            sorted(grouped[model_id], key=lambda entry: int(entry["speaker_id"])),
        )
        for model_id in sorted(grouped)
    ]


def worker_failures(
    entries: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    error: str,
    error_type: str,
) -> list[dict[str, Any]]:
    return [
        _failure_row(
            entry,
            repeat,
            phase="worker_process",
            error_type=error_type,
            error=error,
        )
        for entry in entries
        for repeat in range(repeats)
    ]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def run_worker_job(job_path: Path, output_path: Path) -> int:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if not isinstance(job, Mapping) or job.get("schema") != 1:
            raise ValueError("worker job has an invalid schema")
        entries = job["entries"]
        if not isinstance(entries, list):
            raise ValueError("worker job entries must be a list")
        stimulus = LoudnessStimulus(**job["stimulus"])
        measurements, failures = measure_model(
            entries,
            stimulus=stimulus,
            cache_dir=job.get("cache_dir"),
            offline=bool(job.get("offline", False)),
            repeats=int(job["repeats"]),
        )
        payload = {
            "schema": 1,
            "model_id": str(job["model_id"]),
            "measurements": measurements,
            "failures": failures,
        }
        _write_json_atomic(output_path, payload)
        return 0
    except Exception as exc:
        print(f"worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def run_model_isolated(
    entries: Sequence[Mapping[str, Any]],
    *,
    stimulus: LoudnessStimulus,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    repeats: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one model in a fresh Python process and read its result file."""
    if not entries:
        raise ValueError("isolated model run requires at least one identity")
    model_id = str(entries[0]["model_id"])
    with tempfile.TemporaryDirectory(prefix="pipersynth-loudness-") as directory:
        directory_path = Path(directory)
        job_path = directory_path / "job.json"
        output_path = directory_path / "result.json"
        _write_json_atomic(
            job_path,
            {
                "schema": 1,
                "model_id": model_id,
                "locale": str(entries[0]["locale"]),
                "entries": [dict(entry) for entry in entries],
                "stimulus": asdict(stimulus),
                "repeats": repeats,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "offline": offline,
            },
        )
        command = [
            sys.executable,
            Path(__file__).resolve().as_posix(),
            "--_worker-job",
            str(job_path),
            "--_worker-output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(command, check=False)
        except Exception as exc:
            return [], worker_failures(
                entries,
                repeats=repeats,
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if completed.returncode != 0:
            return [], worker_failures(
                entries,
                repeats=repeats,
                error=f"worker exited with status {completed.returncode}",
                error_type="WorkerProcessError",
            )
        if not output_path.is_file():
            return [], worker_failures(
                entries,
                repeats=repeats,
                error="worker did not produce a result file",
                error_type="WorkerProcessError",
            )
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or payload.get("schema") != 1:
                raise ValueError("worker result has an invalid schema")
            if str(payload.get("model_id")) != model_id:
                raise ValueError("worker result model ID does not match the job")
            measurements = payload.get("measurements")
            failures = payload.get("failures")
            if not isinstance(measurements, list) or not isinstance(failures, list):
                raise ValueError("worker result rows must be lists")
            if not all(isinstance(row, Mapping) for row in [*measurements, *failures]):
                raise ValueError("worker result rows must be objects")
            return [dict(row) for row in measurements], [dict(row) for row in failures]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return [], worker_failures(
                entries,
                repeats=repeats,
                error=f"invalid worker result: {exc}",
                error_type="WorkerResultError",
            )


def aggregate_measurements(
    measurements: Sequence[Mapping[str, Any]], *, repeats: int = 3, max_mad_lu: float = 0.75
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in measurements:
        grouped.setdefault(str(item["calibration_key"]), []).append(item)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        values = [float(item["integrated_lufs"]) for item in items]
        finite = [value for value in values if math.isfinite(value)]
        peaks = [float(item["true_peak_dbtp"]) for item in items]
        finite_peaks = [value for value in peaks if math.isfinite(value)]
        if not finite:
            median_lufs = mad_lu = None
            status = "non_finite_loudness"
        else:
            median_lufs = statistics.median(finite)
            mad_lu = statistics.median(abs(value - median_lufs) for value in finite)
            status = "eligible"
            if len(finite) != len(values) or len(finite_peaks) != len(peaks):
                status = "non_finite_loudness"
            elif len(finite) != repeats:
                status = "insufficient_repeats"
            elif mad_lu > max_mad_lu:
                status = "high_variability"
        rows.append(
            {
                "model_source": items[0]["model_source"],
                "model_id": items[0]["model_id"],
                "quality": items[0]["quality"],
                "speaker_id": items[0]["speaker_id"],
                "speaker_name": items[0].get("speaker_name"),
                "calibration_key": key,
                "locale": items[0]["locale"],
                "samples": len(items),
                "repeat_count": len(items),
                "median_lufs": median_lufs,
                "mad_lu": mad_lu,
                "max_true_peak_dbtp": max(finite_peaks) if finite_peaks else None,
                "status": status,
            }
        )
    return rows


def calibration_candidate(
    aggregate: Mapping[str, Any],
    *,
    reference_lufs: float = -24.0,
    peak_ceiling_dbtp: float = -1.0,
    max_boost_db: float = 8.0,
    max_attenuation_db: float = 12.0,
) -> dict[str, Any]:
    status = str(aggregate["status"])
    if status == "non_finite_loudness":
        return {**dict(aggregate), "status": status}
    measured = float(aggregate["median_lufs"])
    max_peak = float(aggregate["max_true_peak_dbtp"])
    requested = reference_lufs - measured
    safe = peak_ceiling_dbtp - max_peak
    gain = min(requested, safe) if requested > 0 else requested
    status = str(aggregate["status"])
    limited = gain < requested
    if status == "eligible":
        if limited:
            status = "headroom_limited"
        elif requested > max_boost_db:
            status = "review_large_boost"
        elif requested < -max_attenuation_db:
            status = "review_large_attenuation"
    return {
        **dict(aggregate),
        "measured_lufs": measured,
        "reference_lufs": reference_lufs,
        "requested_gain_db": requested,
        "gain_db": gain,
        "max_safe_gain_db": safe,
        "projected_true_peak_dbtp": max_peak + gain,
        "headroom_limited": limited,
        "target_reached": not limited,
        "status": status,
    }


def coverage_report(
    entries: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> dict[str, Any]:
    expected = {str(item["calibration_key"]) for item in entries}
    measured = {
        str(item["calibration_key"]) for item in aggregates if item.get("repeat_count") == repeats
    }
    failed = {str(item["calibration_key"]) for item in failures}
    failed.update(
        str(item["calibration_key"])
        for item in aggregates
        if item.get("status") in {"non_finite_loudness", "insufficient_repeats"}
    )
    failed_expected = failed & expected
    failed_or_missing = (expected - measured) | failed_expected
    return {
        "catalog_identities_expected": len(expected),
        "catalog_identities_measured": len(measured),
        "catalog_identities_failed": len(failed_or_missing),
        "complete": expected == measured and not failed_expected,
    }


def language_preflight_payload(preflight: StimulusPreflight | None) -> dict[str, Any]:
    if preflight is None:
        return {"locales": {}, "supported_count": 0, "unsupported_count": 0}
    locales: dict[str, dict[str, Any]] = {}
    for locale, stimulus in sorted(preflight.stimuli.items()):
        locales[locale] = {
            "status": "supported",
            "generator": stimulus.generator,
            "spoken_text": stimulus.spoken_text,
            "normalized_language": stimulus.normalized_language,
        }
    for locale, error in sorted(preflight.unsupported.items()):
        locales[locale] = {"status": "unsupported", "error": error}
    return {
        "locales": locales,
        "supported_count": len(preflight.stimuli),
        "unsupported_count": len(preflight.unsupported),
    }


def build_report(
    entries: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    policy: BenchmarkPolicy,
    language_preflight: StimulusPreflight | None = None,
) -> dict[str, Any]:
    aggregates = aggregate_measurements(
        measurements, repeats=policy.repeats, max_mad_lu=policy.max_mad_lu
    )
    candidates = [
        calibration_candidate(
            row,
            reference_lufs=policy.reference_lufs,
            peak_ceiling_dbtp=policy.calibration_peak_ceiling_dbtp,
            max_boost_db=policy.max_boost_db,
            max_attenuation_db=policy.max_attenuation_db,
        )
        for row in aggregates
    ]
    return {
        "schema": 2,
        "corpus": policy.name,
        "generated_with": generated_with(),
        "policy": asdict(policy),
        "inventory": list(entries),
        "measurements": list(measurements),
        "aggregates": candidates,
        "failures": list(failures),
        "language_preflight": language_preflight_payload(language_preflight),
        "coverage": coverage_report(entries, aggregates, failures, repeats=policy.repeats),
    }


def _failure_summary_lines(failures: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for failure in failures:
        phase = str(failure.get("phase", "unknown"))
        category = str(failure.get("error_type") or failure.get("error") or "unknown")
        identity = str(
            failure.get(
                "calibration_key",
                f"{failure.get('model_id', 'unknown')}:{failure.get('speaker_id', 'unknown')}",
            )
        )
        grouped.setdefault((phase, category), set()).add(identity)
    if not grouped:
        return ["- none"]
    return [
        f"- {phase}/{category}: {len(identities)} identities"
        for (phase, category), identities in sorted(grouped.items())
    ]


def write_outputs(
    report: Mapping[str, Any], output: Path, *, candidate_path: Path | None = None
) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    (output / "measurements.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = report.get("measurements", [])
        fieldnames = sorted({key for row in rows if isinstance(row, Mapping) for key in row})
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (output / "failures.json").write_text(
        json.dumps(report["failures"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    coverage = report["coverage"]
    preflight = report["language_preflight"]
    unsupported_locales = sorted(
        locale
        for locale, item in preflight["locales"].items()
        if item.get("status") == "unsupported"
    )
    summary = [
        "# PiperSynth voice loudness benchmark",
        "",
        f"- Expected identities: {coverage['catalog_identities_expected']}",
        f"- Measured identities: {coverage['catalog_identities_measured']}",
        f"- Failed identities: {coverage['catalog_identities_failed']}",
        f"- Complete coverage: {coverage['complete']}",
        f"- Distinct locales: {preflight['supported_count'] + preflight['unsupported_count']}",
        f"- Supported stimulus locales: {preflight['supported_count']}",
        f"- Unsupported stimulus locales: {preflight['unsupported_count']}",
        f"- Unsupported: {', '.join(unsupported_locales) if unsupported_locales else 'none'}",
        "",
        "## Failure summary",
        *_failure_summary_lines(report["failures"]),
        "",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    if candidate_path is not None:
        if not coverage["complete"]:
            return False
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    return True


_load_policy = load_policy
_load_fallbacks = load_fallbacks
_entries = inventory_entries
_aggregate = aggregate_measurements
_calibration_candidate = calibration_candidate
_coverage = coverage_report


def print_stimulus_preflight(preflight: StimulusPreflight) -> None:
    print("Language/stimulus preflight")
    print("---------------------------")
    for locale in sorted(set(preflight.stimuli) | set(preflight.unsupported)):
        stimulus = preflight.stimuli.get(locale)
        if stimulus is None:
            print(f"{locale}  unsupported  {preflight.unsupported[locale]}")
        else:
            print(f"{locale}  {stimulus.generator}  {stimulus.spoken_text}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice")
    parser.add_argument("--language")
    parser.add_argument("--locale")
    parser.add_argument("--quality")
    parser.add_argument("--speaker", type=int)
    parser.add_argument("--max-voices", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--list-stimuli", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--write-calibration-candidate", type=Path)
    parser.add_argument("--_worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker_job is not None:
        if args._worker_output is None:
            parser.error("--_worker-output is required with --_worker-job")
        return run_worker_job(args._worker_job, args._worker_output)
    policy = load_policy()
    if args.repeats is not None:
        policy = BenchmarkPolicy(**{**asdict(policy), "repeats": args.repeats})
    manager = VoiceAssetManager(args.cache_dir, offline=args.offline)
    voices = manager.list_voices(refresh=args.refresh_catalog)
    entries = inventory_entries(
        voices,
        voice=args.voice,
        language=args.language,
        locale=args.locale,
        quality=args.quality,
        speaker=args.speaker,
        max_voices=args.max_voices,
    )
    print(f"Catalog cache: {manager.cache_info().directory}")
    print(f"Catalog voices: {len(voices)}; expanded identities: {len(entries)}")
    preflight = preflight_stimuli(entries)
    print_stimulus_preflight(preflight)
    if args.list_stimuli:
        return 0
    failures = stimulus_failures_for_entries(entries, preflight.unsupported)
    if preflight.unsupported and not args.allow_failures:
        report = build_report(entries, [], failures, policy, language_preflight=preflight)
        write_outputs(report, args.output, candidate_path=args.write_calibration_candidate)
        print(
            f"Unsupported stimulus locales: {', '.join(sorted(preflight.unsupported))}. "
            "Use --allow-failures to measure the supported subset."
        )
        return 2
    measurable_entries = [entry for entry in entries if str(entry["locale"]) in preflight.stimuli]
    model_jobs = group_entries_by_model(measurable_entries)
    measurements: list[dict[str, Any]] = []
    for model_index, (model_id, model_entries) in enumerate(model_jobs, 1):
        locale = str(model_entries[0]["locale"])
        print(
            f"[model {model_index}/{len(model_jobs)}] {model_id} ({len(model_entries)} identities)"
        )
        measured, failed = run_model_isolated(
            model_entries,
            stimulus=preflight.stimuli[locale],
            cache_dir=args.cache_dir,
            offline=args.offline,
            repeats=policy.repeats,
        )
        measurements.extend(measured)
        failures.extend(failed)
    report = build_report(entries, measurements, failures, policy, language_preflight=preflight)
    complete = write_outputs(report, args.output, candidate_path=args.write_calibration_candidate)
    if args.write_calibration_candidate and not complete:
        print("Incomplete catalog coverage; production candidate suppressed.")
        return 0 if args.allow_failures else 2
    return 0 if complete or args.allow_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
