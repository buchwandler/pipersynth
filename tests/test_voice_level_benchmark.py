from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.promote_voice_calibration import build_candidate
from benchmarks.voice_level_benchmark import (
    aggregate_measurements,
    build_report,
    expand_identities,
    load_policy,
)
from pipersynth.assets import VoiceMetadata
from pipersynth.voice_level import load_voice_calibration


@pytest.fixture
def metadata() -> VoiceMetadata:
    return VoiceMetadata(
        id="en_US-test-medium",
        name="Test",
        language_code="en_US",
        language_family="en",
        region="US",
        quality="medium",
        num_speakers=2,
        speaker_id_map={"Alex": 0, "Beth": 1},
        aliases=(),
        source_revision="test",
    )


def measurements_for(
    entries: list[dict[str, object]], levels: list[float]
) -> list[dict[str, object]]:
    return [
        {
            **entry,
            "repeat": repeat,
            "sample_rate": 22050,
            "duration_seconds": 1.0,
            "integrated_lufs": level,
        }
        for entry in entries
        for repeat, level in enumerate(levels)
    ]


def report_for(
    entries: list[dict[str, object]], measurements: list[dict[str, object]]
) -> dict[str, object]:
    policy = load_policy()
    return build_report(
        entries,
        measurements,
        [],
        text="A prepared calibration stimulus.",
        language="en_US",
        policy=policy,
    )


def test_expands_every_catalog_speaker_identity(metadata: VoiceMetadata) -> None:
    entries = expand_identities((metadata,))

    assert [entry["speaker_id"] for entry in entries] == [0, 1]
    assert [entry["speaker_name"] for entry in entries] == ["Alex", "Beth"]
    assert entries[0]["calibration_key"] == "piper:en_US-test-medium:medium:speaker-0"


def test_aggregates_repeats_and_builds_runtime_compatible_candidate(
    metadata: VoiceMetadata, tmp_path: Path
) -> None:
    entries = expand_identities((metadata,))
    measurements = measurements_for(entries, [-24.0, -24.1, -23.9])
    policy = load_policy()

    aggregates = aggregate_measurements(measurements, policy)
    assert [row["status"] for row in aggregates] == ["eligible", "eligible"]
    assert aggregates[0]["median_lufs"] == pytest.approx(-24.0)
    assert aggregates[0]["mad_lu"] == pytest.approx(0.1)

    report = report_for(entries, measurements)
    assert report["coverage"] == {
        "catalog_identities_expected": 2,
        "catalog_identities_measured": 2,
        "catalog_identities_failed": 0,
        "failure_count": 0,
        "complete": True,
    }
    candidate, counts = build_candidate(report, allow_partial=False, include_high_variability=False)
    assert counts == {"eligible": 2, "high_variability": 0, "incomplete": 0}
    output = tmp_path / "candidate.json"
    output.write_text(json.dumps(candidate), encoding="utf-8")
    catalog = load_voice_calibration(output)
    assert len(catalog.voices) == 2
    assert catalog.corpus == "pipersynth-prepared-speech-v1"


def test_partial_coverage_requires_explicit_promotion(metadata: VoiceMetadata) -> None:
    entries = expand_identities((metadata,))
    measurements = measurements_for(entries[:1], [-24.0, -24.1, -23.9])
    report = report_for(entries, measurements)

    with pytest.raises(ValueError, match="allow-partial"):
        build_candidate(report, allow_partial=False, include_high_variability=False)
    candidate, _ = build_candidate(report, allow_partial=True, include_high_variability=False)
    assert len(candidate["voices"]) == 1


def test_high_variability_is_excluded_without_explicit_review(metadata: VoiceMetadata) -> None:
    entries = expand_identities((metadata,))[:1]
    measurements = measurements_for(entries, [-20.0, -21.0, -23.0])
    report = report_for(entries, measurements)

    with pytest.raises(ValueError, match="no eligible"):
        build_candidate(report, allow_partial=False, include_high_variability=False)
    candidate, counts = build_candidate(report, allow_partial=False, include_high_variability=True)
    assert counts["high_variability"] == 1
    assert len(candidate["voices"]) == 1


def test_identity_override_preserves_retained_gain_without_weakening_global_floor() -> None:
    policy = load_policy()
    override_key = "piper:en_US-libritts_r-medium:medium:speaker-761"
    unrelated_key = "piper:en_US-test-medium:medium:speaker-0"
    retained_catalog = json.loads(
        (
            Path(__file__).resolve().parents[1] / "pipersynth/data/voice_level_calibration.json"
        ).read_text(
            encoding="utf-8",
        )
    )
    retained_gain = retained_catalog["voices"][override_key]["gain_db"]
    assert policy["min_gain_db"] == -12.0
    assert policy["identity_overrides"][override_key]["min_gain_db"] == retained_gain
    assert policy["identity_overrides"][override_key]["rationale"]
    entries = [
        {
            "model_source": "piper",
            "model_id": "en_US-libritts_r-medium",
            "quality": "medium",
            "locale": "en_US",
            "speaker_id": 761,
            "speaker_name": None,
            "calibration_key": override_key,
        },
        {
            "model_source": "piper",
            "model_id": "en_US-test-medium",
            "quality": "medium",
            "locale": "en_US",
            "speaker_id": 0,
            "speaker_name": None,
            "calibration_key": unrelated_key,
        },
    ]
    measured_lufs = policy["reference_lufs"] - retained_gain
    report = report_for(entries, measurements_for(entries, [measured_lufs] * 3))
    gains = {row["calibration_key"]: row["gain_db"] for row in report["aggregates"]}
    assert gains[override_key] == pytest.approx(retained_gain)
    assert gains[unrelated_key] == -12.0
    candidate, _ = build_candidate(report, allow_partial=False, include_high_variability=False)
    assert candidate["voices"][override_key]["gain_db"] == pytest.approx(retained_gain)
    assert candidate["voices"][unrelated_key]["gain_db"] == -12.0
