import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _model_entries(model_id="model"):
    return [
        {
            "model_source": "piper",
            "model_id": model_id,
            "quality": "medium",
            "speaker_id": speaker_id,
            "speaker_name": None,
            "calibration_key": f"piper:{model_id}:medium:speaker-{speaker_id}",
            "locale": "en_US",
        }
        for speaker_id in (0, 1)
    ]


def _stimulus():
    return benchmark.LoudnessStimulus(
        "en_US", "en", benchmark.COUNT_SOURCE, "one, two.", "test", False
    )


def test_group_entries_by_model_is_deterministic_and_preserves_speakers():
    entries = [
        {"model_id": "b", "speaker_id": 0},
        {"model_id": "a", "speaker_id": 1},
        {"model_id": "a", "speaker_id": 0},
    ]
    grouped = benchmark.group_entries_by_model(entries)
    assert [(model_id, [item["speaker_id"] for item in items]) for model_id, items in grouped] == [
        ("a", [0, 1]),
        ("b", [0]),
    ]


def test_measure_model_reuses_one_pipeline_and_overrides_speakers(monkeypatch):
    opened = []
    runs = []
    closed = []

    class FakePipeline:
        def run(self, text, *, speaker):
            runs.append((text, speaker))
            return SimpleNamespace(audio=np.ones(100, dtype=np.float32), sample_rate=22050)

        def close(self):
            closed.append(True)

    def open_pipeline(model_id, **kwargs):
        opened.append((model_id, kwargs))
        return FakePipeline()

    monkeypatch.setattr(benchmark.PiperPipeline, "from_pretrained", staticmethod(open_pipeline))
    monkeypatch.setattr(
        benchmark,
        "measure_loudness",
        lambda audio, sample_rate: SimpleNamespace(
            integrated_lufs=-20.0, sample_peak_dbfs=-3.0, true_peak_dbtp=-3.0
        ),
    )

    measurements, failures = benchmark.measure_model(
        _model_entries(), stimulus=_stimulus(), repeats=2
    )

    assert not failures
    assert len(opened) == 1
    assert len(closed) == 1
    assert [speaker for _, speaker in runs] == [0, 0, 1, 1]
    assert opened[0][1]["language_policy"] == "allow"
    assert opened[0][1]["loudness"].voice_leveling == "off"
    assert opened[0][1]["loudness"].target_lufs is None
    assert opened[0][1]["generation"].normalize_audio is True
    assert opened[0][1]["generation"].volume == 1.0
    assert len(measurements) == 4


def test_run_model_isolated_launches_one_process_per_model(monkeypatch):
    commands = []
    failures = []

    def launch(command, check):
        commands.append(command)
        output = Path(command[command.index("--_worker-output") + 1])
        output.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "model_id": json.loads(
                        Path(command[command.index("--_worker-job") + 1]).read_text()
                    )["model_id"],
                    "measurements": [],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", launch)
    for _model_id, entries in benchmark.group_entries_by_model(
        _model_entries("a") + _model_entries("b")
    ):
        _, model_failures = benchmark.run_model_isolated(entries, stimulus=_stimulus(), repeats=1)
        failures.extend(model_failures)

    assert not failures
    assert len(commands) == 2
    assert all(command[1].endswith("benchmarks/voice_loudness.py") for command in commands)


def test_malformed_worker_result_fails_every_expected_repeat(monkeypatch):
    def launch(command, check):
        output = Path(command[command.index("--_worker-output") + 1])
        output.write_text("not json", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", launch)
    _, failures = benchmark.run_model_isolated(_model_entries(), stimulus=_stimulus(), repeats=3)
    assert len(failures) == 6
    assert {item["phase"] for item in failures} == {"worker_process"}
    assert {item["error_type"] for item in failures} == {"WorkerResultError"}


def test_worker_crash_fails_every_expected_repeat(monkeypatch):
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda command, check: SimpleNamespace(returncode=7)
    )
    _, failures = benchmark.run_model_isolated(_model_entries(), stimulus=_stimulus(), repeats=3)
    assert len(failures) == 6
    assert {item["phase"] for item in failures} == {"worker_process"}
    assert {item["error_type"] for item in failures} == {"WorkerProcessError"}


def test_nonfinite_measurement_becomes_json_safe_failure(monkeypatch, tmp_path):
    class FakePipeline:
        def run(self, text, *, speaker):
            return SimpleNamespace(audio=np.ones(10, dtype=np.float32), sample_rate=22050)

        def close(self):
            pass

    monkeypatch.setattr(
        benchmark.PiperPipeline,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakePipeline()),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_loudness",
        lambda audio, sample_rate: SimpleNamespace(
            integrated_lufs=float("nan"), sample_peak_dbfs=-3.0, true_peak_dbtp=-3.0
        ),
    )
    entries = _model_entries()[:1]
    measurements, failures = benchmark.measure_model(entries, stimulus=_stimulus(), repeats=1)
    report = benchmark.build_report(entries, measurements, failures, benchmark.load_policy())

    assert measurements == []
    assert len(failures) == 1
    assert failures[0]["phase"] == "measurement"
    benchmark.write_outputs(report, tmp_path)
    assert "NaN" not in (tmp_path / "measurements.json").read_text()
    json.loads((tmp_path / "measurements.json").read_text())


def test_spokenform_canonical_language_is_recorded(monkeypatch):
    monkeypatch.setattr(
        benchmark, "_spoken_count_words", lambda locale: ("canonical", ("one", "two"))
    )
    stimulus = benchmark.resolve_count_stimulus("regional", {})
    assert stimulus.normalized_language == "canonical"


def test_summary_lists_only_unsupported_locales_and_provenance(tmp_path):
    entries = [
        {"calibration_key": "supported", "locale": "en_US"},
        {"calibration_key": "unsupported", "locale": "bg_BG"},
    ]
    preflight = benchmark.StimulusPreflight(
        stimuli={"en_US": _stimulus()}, unsupported={"bg_BG": "missing"}
    )
    failures = benchmark.stimulus_failures_for_entries(entries, {"bg_BG": "missing"})
    report = benchmark.build_report(
        entries, [], failures, benchmark.load_policy(), language_preflight=preflight
    )
    benchmark.write_outputs(report, tmp_path)
    summary = (tmp_path / "summary.md").read_text()
    unsupported_line = next(
        line for line in summary.splitlines() if line.startswith("- Unsupported:")
    )
    assert unsupported_line == "- Unsupported: bg_BG"
    assert "## Failure summary" in summary
    assert report["generated_with"]["spokenform"]
    assert report["generated_with"]["piperg2p"]
