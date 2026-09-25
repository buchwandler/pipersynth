from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unbuilt_source_uses_unknown_version_fallback(tmp_path: Path) -> None:
    package_dir = tmp_path / "pipersynth"
    shutil.copytree(
        PROJECT_ROOT / "pipersynth",
        package_dir,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (package_dir / "_version.py").unlink()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """\
import importlib.machinery
import sys
class BlockGeneratedVersion:
    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname == "pipersynth._version":
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(
    sys.meta_path.index(importlib.machinery.PathFinder) + 1, BlockGeneratedVersion
)
import pipersynth
print(pipersynth.__version__)
print(pipersynth.__version_tuple__)
            """,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["0.0.0+unknown", "(0, 0, 0)"]
