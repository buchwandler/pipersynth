"""Install a built wheel into a fresh environment and probe its public API."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Any

_API_PROBE = r"""
import importlib.abc
import json
import socket
import sys
from dataclasses import asdict

class BlockRuntimeAndNetwork(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnxruntime" or fullname.startswith("onnxruntime."):
            raise AssertionError(f"API smoke probe imported {fullname}")
        return None

sys.meta_path.insert(0, BlockRuntimeAndNetwork())
def blocked_connect(*args, **kwargs):
    raise AssertionError("API smoke probe attempted network access")
socket.socket.connect = blocked_connect

import pipersynth

required = (
    "PiperVoice",
    "VoiceAssetManager",
    "SynthesisRequest",
    "SynthesisResult",
    "SynthesisConfig",
    "LinguisticToken",
    "PronunciationOverride",
    "VoiceLevelConfig",
    "SynthesisInputTooLongError",
)
assert all(getattr(pipersynth, name, None) is not None for name in required)
assert callable(pipersynth.PiperVoice.synthesize)
assert callable(pipersynth.VoiceAssetManager)
request = pipersynth.SynthesisRequest("wheel-smoke", "Hello", "en-us")
config = pipersynth.SynthesisConfig()
assert request.text == "Hello"
assert config.voice_level is not None
contract = pipersynth.request_api_contract()
assert contract.version == 1
assert not any(name == "onnxruntime" or name.startswith("onnxruntime.") for name in sys.modules)
print(json.dumps({"exports": sorted(pipersynth.__all__), "contract": asdict(contract)}))
"""


def _probe(python: Path, *, project_root: Path, isolated: bool) -> dict[str, Any]:
    command = [str(python)]
    if isolated:
        command.append("-I")
    command.extend(["-c", _API_PROBE])
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare source and installed-wheel public API surfaces."
    )
    parser.add_argument("wheel_directory", type=Path)
    parser.add_argument(
        "--system-site-packages",
        action="store_true",
        help="reuse host dependencies for local platforms without installable dependency wheels",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="install only PiperSynth (requires --system-site-packages)",
    )
    args = parser.parse_args()
    if args.no_deps and not args.system_site_packages:
        parser.error("--no-deps requires --system-site-packages")

    wheel_directory = args.wheel_directory.resolve()
    wheels = sorted(wheel_directory.glob("pipersynth-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"expected exactly one PiperSynth wheel in {wheel_directory}, found {wheels}"
        )
    wheel = wheels[0]
    project_root = Path(__file__).resolve().parents[1]

    source_api = _probe(Path(sys.executable), project_root=project_root, isolated=False)
    with tempfile.TemporaryDirectory(prefix="pipersynth-wheel-api-") as temporary_directory:
        environment = Path(temporary_directory) / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=args.system_site_packages).create(
            environment
        )
        wheel_python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(wheel_python),
                "-m",
                "pip",
                "install",
                *(["--no-deps"] if args.no_deps else []),
                "--disable-pip-version-check",
                str(wheel),
            ],
            check=True,
        )
        wheel_api = _probe(wheel_python, project_root=project_root, isolated=True)

    if source_api != wheel_api:
        raise SystemExit("installed-wheel API does not match the source API")
    print(f"Wheel API smoke check passed: {wheel.name}")


if __name__ == "__main__":
    main()
