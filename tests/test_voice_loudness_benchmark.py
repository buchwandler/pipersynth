import pytest

from benchmarks.voice_loudness import (
    aggregate_measurements,
    calibration_candidate,
    inventory_entries,
    resolve_count_stimulus,
)
from pipersynth.assets import VoiceMetadata


def _voice(num_speakers=2):
    return VoiceMetadata(
        "voice",
        "Voice",
        "en_US",
        "en",
        None,
        "medium",
        num_speakers,
        {"alice": 0, "bob": 1},
        (),
        "revision",
    )


def test_inventory_expands_every_numeric_speaker_and_alias_filter():
    entries = inventory_entries([_voice()])
    assert [entry["speaker_id"] for entry in entries] == [0, 1]
    assert inventory_entries([_voice()], voice="voice", speaker=1)[0]["calibration_key"].endswith(
        "speaker-1"
    )


def test_aggregation_uses_median_mad_and_safe_boost():
    entries = inventory_entries([_voice(1)])
    measurements = [
        {**entries[0], "integrated_lufs": value, "true_peak_dbtp": -3.0, "repeat": index}
        for index, value in enumerate((-20.0, -20.2, -19.8))
    ]
    aggregate = aggregate_measurements(measurements)[0]
    assert aggregate["median_lufs"] == -20.0
    assert aggregate["mad_lu"] == pytest.approx(0.2)
    candidate = calibration_candidate(aggregate)
    assert candidate["requested_gain_db"] == -4.0
    assert candidate["gain_db"] == -4.0


def test_positive_boost_is_headroom_limited_and_fallback_rejects_digits():
    entries = inventory_entries([_voice(1)])
    aggregate = aggregate_measurements(
        [
            {**entries[0], "integrated_lufs": -30.0, "true_peak_dbtp": -0.5, "repeat": i}
            for i in range(3)
        ]
    )[0]
    candidate = calibration_candidate(aggregate)
    assert candidate["gain_db"] == -0.5
    assert candidate["status"] == "headroom_limited"
    stimulus = resolve_count_stimulus(
        "xx",
        fallbacks={"xx": {"text": "one, two, three, four, five, six, seven, eight, nine, ten."}},
    )
    assert stimulus.fallback_used
