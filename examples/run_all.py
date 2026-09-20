"""Run and validate the maintained PiperSynth examples."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

from utterplan import UtterancePlan

try:
    from examples._output import ARTEFACT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from _output import ARTEFACT_DIR, PROJECT_ROOT

_OUTPUT_ENV = "PIPERSYNTH_EXAMPLE_OUTPUT_DIR"
_EXCLUDED_FILES = {"__init__.py", "_output.py", "run_all.py"}
_OPTIONAL_EXAMPLES = {"german.py", "homographs.py"}
RunCommand = Callable[..., subprocess.CompletedProcess[str]]
_RESOURCE_HEAVY_EXAMPLES = {"all_voices.py"}


def _example_paths(
    *, include_optional: bool = False, include_resource_heavy: bool = False
) -> list[Path]:
    paths = []
    for path in sorted(PROJECT_ROOT.joinpath("examples").glob("*.py")):
        if path.name in _EXCLUDED_FILES:
            continue
        if not include_optional and path.name in _OPTIONAL_EXAMPLES:
            continue
        if not include_resource_heavy and path.name in _RESOURCE_HEAVY_EXAMPLES:
            continue
        paths.append(path)
    return paths


def _label(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _validate_plan(path: Path) -> None:
    plan = UtterancePlan.load(path)
    plan.validate()
    if not plan.plan_id:
        raise RuntimeError(f"{path} has no plan_id")
    if not plan.units:
        raise RuntimeError(f"{path} has no render units")


def _validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise RuntimeError(f"{path} must be mono, got {stream.getnchannels()} channels")
        if stream.getsampwidth() != 2:
            raise RuntimeError(
                f"{path} must be 16-bit PCM, got sample width {stream.getsampwidth()}"
            )
        if stream.getframerate() <= 0:
            raise RuntimeError(f"{path} has invalid sample rate")
        if stream.getnframes() <= 0:
            raise RuntimeError(f"{path} contains no audio frames")


def _validate_artefacts(output_dir: Path) -> None:
    plans = sorted(output_dir.rglob("*.utterplan.json"))
    wavs = sorted(output_dir.rglob("*.wav"))

    if not plans:
        raise RuntimeError("example produced no .utterplan.json artefact")
    if not wavs:
        raise RuntimeError("example produced no .wav artefact")

    for path in plans:
        _validate_plan(path)
    for path in wavs:
        _validate_wav(path)


def run_examples(
    paths: Sequence[Path],
    *,
    runner: RunCommand = subprocess.run,
    fail_fast: bool = False,
) -> int:
    failures = 0

    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)
        output_name = "__".join(relative.with_suffix("").parts)
        output_dir = ARTEFACT_DIR / output_name

        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        environment = os.environ.copy()
        environment[_OUTPUT_ENV] = str(output_dir)

        print(f"\n=== {_label(path)} ===")
        result = runner(
            [sys.executable, str(relative)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            text=True,
        )

        error: Exception | None = None
        if result.returncode:
            error = RuntimeError(f"process exited with {result.returncode}")
        else:
            try:
                _validate_artefacts(output_dir)
            except Exception as exc:
                error = exc

        if error is None:
            print(f"PASSED: {_label(path)}")
            continue

        failures += 1
        print(f"FAILED: {_label(path)}: {error}")
        if fail_fast:
            break

    passed = len(paths) - failures
    print(f"\nCompleted {len(paths)} examples: {passed} passed, {failures} failed.")
    print(f"Artefacts: {ARTEFACT_DIR}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="list selected examples without running them",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="include slower or additional-model examples",
    )
    parser.add_argument(
        "--include-resource-heavy",
        action="store_true",
        help="include examples that may download or synthesize the full catalog",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed example",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _example_paths(include_optional=args.include_optional)
    paths = _example_paths(
        include_optional=args.include_optional,
        include_resource_heavy=args.include_resource_heavy,
    )
    if args.list:
        for path in paths:
            print(_label(path))
        return 0

    return min(
        run_examples(paths, fail_fast=args.fail_fast),
        1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
