#!/usr/bin/env python3
"""Run selected PiperVoice examples in isolated output directories."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import wave
from collections.abc import Sequence
from pathlib import Path

try:
    from examples._output import ARTEFACT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from _output import ARTEFACT_DIR, PROJECT_ROOT

_OUTPUT_ENV = "PIPERSYNTH_EXAMPLE_OUTPUT_DIR"
_EXCLUDED = {"__init__.py", "_output.py", "all_voices.py", "run_all.py"}
_OPTIONAL = {"german.py", "homographs.py"}


def example_paths(*, include_optional: bool = False) -> list[Path]:
    return [
        path
        for path in sorted(PROJECT_ROOT.joinpath("examples").glob("*.py"))
        if path.name not in _EXCLUDED and (include_optional or path.name not in _OPTIONAL)
    ]


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise RuntimeError(f"{path} must be mono 16-bit PCM")
        if stream.getframerate() <= 0 or stream.getnframes() <= 0:
            raise RuntimeError(f"{path} has no valid audio frames")


def run_examples(paths: Sequence[Path], *, fail_fast: bool = False) -> int:
    failures = 0
    for path in paths:
        output_dir = ARTEFACT_DIR / path.stem
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment[_OUTPUT_ENV] = str(output_dir)
        relative = path.relative_to(PROJECT_ROOT)
        print(f"\n=== {relative} ===")
        result = subprocess.run(
            [sys.executable, str(relative)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            text=True,
        )
        wavs = sorted(output_dir.rglob("*.wav"))
        if result.returncode == 0 and wavs:
            try:
                for wav in wavs:
                    validate_wav(wav)
            except Exception as error:
                print(f"FAILED: {relative}: {error}")
                failures += 1
            else:
                print(f"PASSED: {relative}")
        else:
            print(f"FAILED: {relative}: process={result.returncode}, wavs={len(wavs)}")
            failures += 1
        if failures and fail_fast:
            break
    print(f"\nCompleted {len(paths)} examples with {failures} failure(s).")
    print(f"Artefacts: {ARTEFACT_DIR}")
    return min(failures, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = example_paths(include_optional=args.include_optional)
    if args.list:
        for path in paths:
            print(path.relative_to(PROJECT_ROOT))
        return 0
    return run_examples(paths, fail_fast=args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())
