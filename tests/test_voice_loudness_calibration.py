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


def _partial_report():
    voices = [
        VoiceMetadata("aaa", "Voice", "en_US", "en", None, "medium", 1, {}, (), "rev"),
        VoiceMetadata("zzz", "Missing", "en_US", "en", None, "medium", 1, {}, (), "rev"),
    ]
    entries = inventory_entries(voices)
    measurements = [
        {**entries[0], "repeat": index, "integrated_lufs": -20.0, "true_peak_dbtp": -3.0}
        for index in range(3)
    ]
    failures = [{"calibration_key": entries[1]["calibration_key"]}]
    return build_report(entries, measurements, failures, load_policy())


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


def test_partial_promotion_requires_explicit_opt_in():
    report = _partial_report()
    with pytest.raises(PromotionError):
        build_runtime_calibration(report)
    catalog = build_runtime_calibration(report, require_complete_coverage=False)
    assert list(catalog["voices"]) == ["piper:aaa:medium:speaker-0"]


def test_explicit_high_variability_status_is_promotable():
    report = _report()
    report["aggregates"][0]["status"] = "high_variability"
    report["aggregates"][0]["mad_lu"] = 1.0
    assert build_runtime_calibration(report)["voices"] == {}
    catalog = build_runtime_calibration(report, reviewed_statuses=frozenset({"high_variability"}))
    assert "piper:voice:medium:speaker-0" in catalog["voices"]


def test_headroom_limited_gain_stays_peak_safe():
    report = _report()
    aggregate = report["aggregates"][0]
    aggregate.update(status="headroom_limited", median_lufs=-25.0, max_true_peak_dbtp=-0.5)
    catalog = build_runtime_calibration(report, reviewed_statuses=frozenset({"headroom_limited"}))
    assert catalog["voices"]["piper:voice:medium:speaker-0"]["gain_db"] == pytest.approx(-0.5)


def test_large_attenuation_is_not_clamped():
    report = _report()
    aggregate = report["aggregates"][0]
    aggregate.update(status="review_large_attenuation", median_lufs=-11.0, max_true_peak_dbtp=-3.0)
    catalog = build_runtime_calibration(
        report, reviewed_statuses=frozenset({"review_large_attenuation"})
    )
    assert catalog["voices"]["piper:voice:medium:speaker-0"]["gain_db"] == pytest.approx(-13.0)


def test_explicit_review_does_not_promote_insufficient_repeats():
    report = _report()
    report["aggregates"][0]["status"] = "high_variability"
    report["aggregates"][0]["repeat_count"] = 2
    catalog = build_runtime_calibration(report, reviewed_statuses=frozenset({"high_variability"}))
    assert catalog["voices"] == {}


@pytest.mark.parametrize("field", ["median_lufs", "mad_lu", "max_true_peak_dbtp"])
def test_partial_mode_rejects_nonfinite_measurements(field):
    report = _partial_report()
    report["aggregates"][0][field] = float("nan")
    with pytest.raises(PromotionError):
        build_runtime_calibration(report, require_complete_coverage=False)


def test_partial_mode_rejects_invalid_coverage_identity_counts():
    report = _partial_report()
    report["coverage"]["catalog_identities_failed"] = 0
    with pytest.raises(PromotionError):
        build_runtime_calibration(report, require_complete_coverage=False)


def test_partial_mode_rejects_invalid_identity_schema_and_duplicates():
    malformed = _partial_report()
    malformed["aggregates"][0]["speaker_id"] = "0"
    with pytest.raises(PromotionError):
        build_runtime_calibration(malformed, require_complete_coverage=False)
    duplicate = _partial_report()
    duplicate["aggregates"].append(deepcopy(duplicate["aggregates"][0]))
    with pytest.raises(PromotionError):
        build_runtime_calibration(duplicate, require_complete_coverage=False)
    wrong_schema = _partial_report()
    wrong_schema["schema"] = 1
    with pytest.raises(PromotionError):
        build_runtime_calibration(wrong_schema, require_complete_coverage=False)
    missing_provenance = _partial_report()
    missing_provenance["generated_with"] = {}
    with pytest.raises(PromotionError):
        build_runtime_calibration(missing_provenance, require_complete_coverage=False)


def test_production_path_is_never_an_automatic_output_target():
    assert PRODUCTION_CALIBRATION_PATH.name == "voice_level_calibration.json"
