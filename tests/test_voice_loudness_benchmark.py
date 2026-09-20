import inspect
from collections.abc import Mapping

import pytest

import benchmarks.voice_loudness as benchmark
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


def test_preflight_resolves_each_distinct_locale_once(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        {"locale": "de_DE", "calibration_key": "de"},
        {"locale": "en_US", "calibration_key": "en"},
        {"locale": "en_US", "calibration_key": "en-2"},
    ]
    calls: list[str] = []
    stimulus = benchmark.LoudnessStimulus("locale", "en", "source", "one", "test", False)

    def resolve(
        locale: str, _fallbacks: Mapping[str, Mapping[str, object]]
    ) -> benchmark.LoudnessStimulus:
        calls.append(locale)
        return benchmark.LoudnessStimulus(locale, "language", "source", "one", "test", False)

    monkeypatch.setattr(benchmark, "resolve_count_stimulus", resolve)
    preflight = benchmark.preflight_stimuli(entries, {})
    assert calls == ["de_DE", "en_US"]
    assert set(preflight.stimuli) == {"de_DE", "en_US"}
    assert stimulus.locale == "locale"


def test_preflight_collects_all_unsupported_locales_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [{"locale": "bg_BG"}, {"locale": "en_US"}, {"locale": "de_DE"}]

    def resolve(
        locale: str, _fallbacks: Mapping[str, Mapping[str, object]]
    ) -> benchmark.LoudnessStimulus:
        if locale == "bg_BG":
            raise benchmark.StimulusResolutionError("missing bg_BG")
        return benchmark.LoudnessStimulus(locale, "language", "source", "one", "test", False)

    monkeypatch.setattr(benchmark, "resolve_count_stimulus", resolve)
    preflight = benchmark.preflight_stimuli(entries, {})
    assert list(preflight.unsupported) == ["bg_BG"]
    assert set(preflight.stimuli) == {"de_DE", "en_US"}


def test_measure_repeats_requires_pre_resolved_stimulus() -> None:
    parameter = inspect.signature(benchmark.measure_repeats).parameters["stimulus"]
    assert parameter.default is inspect.Parameter.empty


def test_unsupported_locale_blocks_every_matching_identity() -> None:
    entries = [
        {"locale": "bg_BG", "calibration_key": "one"},
        {"locale": "bg_BG", "calibration_key": "two"},
    ]
    failures = benchmark.stimulus_failures_for_entries(entries, {"bg_BG": "unsupported"})
    assert [item["calibration_key"] for item in failures] == ["one", "two"]
    assert all(item["status"] == "unsupported_stimulus" for item in failures)


def test_coverage_does_not_double_count_explicit_failures() -> None:
    entries = [{"calibration_key": "one"}, {"calibration_key": "two"}]
    aggregates = [{"calibration_key": "one", "repeat_count": 3}]
    report = benchmark.coverage_report(entries, aggregates, [{"calibration_key": "two"}], repeats=3)
    assert report["catalog_identities_failed"] == 1
    assert report["complete"] is False
