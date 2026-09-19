from copy import deepcopy

import pytest

from benchmarks.voice_loudness import build_report, inventory_entries, load_policy
from benchmarks.voice_loudness_calibration import (
    PRODUCTION_CALIBRATION_PATH,
    PromotionError,
    build_runtime_calibration,
)
from pipersynth.assets import VoiceMetadata


def _report():
    voice = VoiceMetadata("voice", "Voice", "en_US", "en", None, "medium", 1, {}, (), "rev")
    entry = inventory_entries([voice])[0]
    measurements = [
        {**entry, "repeat": index, "integrated_lufs": -20.0, "true_peak_dbtp": -3.0}
        for index in range(3)
    ]
    return build_report([entry], measurements, [], load_policy())


def test_promotion_recomputes_gain_and_strips_benchmark_fields():
    catalog = build_runtime_calibration(_report())
    record = catalog["voices"]["piper:voice:medium:speaker-0"]
    assert record["gain_db"] == -4.0
    assert "requested_gain_db" not in record
    assert set(record) == {
        "gain_db",
        "measured_lufs",
        "reference_lufs",
        "mad_lu",
        "samples",
        "method",
        "corpus_version",
    }


def test_promotion_rejects_incomplete_coverage_and_nonfinite_rows():
    report = _report()
    incomplete = deepcopy(report)
    incomplete["coverage"]["complete"] = False
    with pytest.raises(PromotionError):
        build_runtime_calibration(incomplete)
    nonfinite = _report()
    nonfinite["aggregates"][0]["median_lufs"] = float("nan")
    with pytest.raises(PromotionError):
        build_runtime_calibration(nonfinite)


def test_production_path_is_never_an_automatic_output_target():
    assert PRODUCTION_CALIBRATION_PATH.name == "voice_level_calibration.json"
