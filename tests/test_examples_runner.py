from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Literal

import pytest
from utterplan import PlannerConfig, UtterancePlanner

import examples.run_all as runner_module


def _write_plan(path: Path) -> None:
    plan = UtterancePlanner(
        PlannerConfig(language="en-us", document_format="plain", text_preparation="identity")
    ).plan("Hello.", unit="sentence")
    plan.save(path)


def _write_wav(
    path: Path, *, channels: int = 1, sample_width: int = 2, frames: bytes = b"\x00\x00"
) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(22050)
        stream.writeframes(frames)


def _example_tree(tmp_path: Path, *names: str) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    for name in names:
        (examples / name).touch()


def _fake_runner(
    mode: Literal[
        "valid", "no-plan", "no-wav", "malformed-plan", "invalid-wav", "failed"
    ] = "valid",
    calls: list[dict[str, object]] | None = None,
):
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(kwargs)
        output_dir = Path(kwargs["env"]["PIPERSYNTH_EXAMPLE_OUTPUT_DIR"])  # type: ignore[index]
        if mode == "failed":
            return subprocess.CompletedProcess(command, 3)
        if mode == "valid":
            _write_plan(output_dir / "example.utterplan.json")
            _write_wav(output_dir / "example.wav")
        elif mode == "no-wav":
            _write_plan(output_dir / "example.utterplan.json")
        elif mode == "malformed-plan":
            (output_dir / "example.utterplan.json").write_text("{}")
            _write_wav(output_dir / "example.wav")
        elif mode == "invalid-wav":
            _write_plan(output_dir / "example.utterplan.json")
            _write_wav(output_dir / "example.wav", frames=b"")
        return subprocess.CompletedProcess(command, 0)

    return run


def test_example_paths_exclude_helpers_and_optional_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _example_tree(tmp_path, "basic.py", "german.py", "homographs.py", "_output.py", "run_all.py")
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)

    assert [path.name for path in runner_module._example_paths()] == ["basic.py"]
    assert [path.name for path in runner_module._example_paths(include_optional=True)] == [
        "basic.py",
        "german.py",
        "homographs.py",
    ]


def test_run_examples_injects_output_directory_and_validates_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _example_tree(tmp_path, "basic.py")
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)
    artefacts = tmp_path / "artefacts"
    monkeypatch.setattr(runner_module, "ARTEFACT_DIR", artefacts)
    stale = artefacts / "examples__basic" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")
    calls: list[dict[str, object]] = []

    assert (
        runner_module.run_examples(
            [tmp_path / "examples/basic.py"],
            runner=_fake_runner(calls=calls),
        )
        == 0
    )
    assert calls[0]["env"]["PIPERSYNTH_EXAMPLE_OUTPUT_DIR"] == str(artefacts / "examples__basic")  # type: ignore[index]
    assert not stale.exists()


@pytest.mark.parametrize("mode", ["failed", "no-plan", "no-wav", "malformed-plan", "invalid-wav"])
def test_run_examples_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _example_tree(tmp_path, "basic.py")
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "ARTEFACT_DIR", tmp_path / "artefacts")

    assert (
        runner_module.run_examples(
            [tmp_path / "examples/basic.py"],
            runner=_fake_runner(mode=mode),  # type: ignore[arg-type]
        )
        == 1
    )


def test_run_examples_accepts_valid_plan_and_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _example_tree(tmp_path, "basic.py")
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "ARTEFACT_DIR", tmp_path / "artefacts")

    assert (
        runner_module.run_examples(
            [tmp_path / "examples/basic.py"],
            runner=_fake_runner(),
        )
        == 0
    )


def test_run_examples_fail_fast_stops_after_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _example_tree(tmp_path, "first.py", "second.py")
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "ARTEFACT_DIR", tmp_path / "artefacts")
    calls: list[dict[str, object]] = []

    assert (
        runner_module.run_examples(
            [tmp_path / "examples/first.py", tmp_path / "examples/second.py"],
            runner=_fake_runner(mode="failed", calls=calls),
            fail_fast=True,
        )
        == 1
    )
    assert len(calls) == 1
