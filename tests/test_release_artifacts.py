from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.check_release_artifacts import CORE_MINIMUMS, validate_release_artifacts


def _write_artifacts(
    directory: Path,
    *,
    requirements: list[str] | None = None,
    private_sdist_path: str | None = None,
) -> None:
    version = "0.2.0"
    requirements = requirements or [f"{name}>={minimum}" for name, minimum in CORE_MINIMUMS.items()]
    wheel = directory / f"pipersynth-{version}-py3-none-any.whl"
    metadata = "\n".join(
        ["Metadata-Version: 2.1", "Name: pipersynth", f"Version: {version}"]
        + [f"Requires-Dist: {requirement}" for requirement in requirements]
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pipersynth-0.2.0.dist-info/METADATA", metadata)
        archive.writestr("pipersynth/py.typed", "")
        archive.writestr("pipersynth/data/voice_level_calibration.json", "{}")

    sdist = directory / f"pipersynth-{version}.tar.gz"
    pkg_info_path = f"pipersynth-{version}/PKG-INFO"
    with tarfile.open(sdist, "w:gz") as archive:
        pkg_info = tarfile.TarInfo(pkg_info_path)
        content = f"Metadata-Version: 2.1\nName: pipersynth\nVersion: {version}\n".encode()
        pkg_info.size = len(content)
        archive.addfile(pkg_info, io.BytesIO(content))
        if private_sdist_path is not None:
            private_file = tarfile.TarInfo(f"pipersynth-{version}/{private_sdist_path}")
            private_file.size = 0
            archive.addfile(private_file, io.BytesIO(b""))


def test_validates_current_release_artifacts(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    validate_release_artifacts(tmp_path, "v0.2.0")


@pytest.mark.parametrize("dependency", ["utterplan", "audiocompose", "ssmd", "phrasplit"])
def test_rejects_forbidden_runtime_dependencies(tmp_path: Path, dependency: str) -> None:
    requirements = [f"{name}>={minimum}" for name, minimum in CORE_MINIMUMS.items()]
    _write_artifacts(tmp_path, requirements=[*requirements, f"{dependency}>=0.1"])

    with pytest.raises(ValueError, match="removed or unused dependencies"):
        validate_release_artifacts(tmp_path, "v0.2.0")


def test_rejects_missing_core_dependency_floor(tmp_path: Path) -> None:
    requirements = [
        f"{name}>={minimum}" for name, minimum in CORE_MINIMUMS.items() if name != "piperg2p"
    ]
    _write_artifacts(tmp_path, requirements=requirements)

    with pytest.raises(ValueError, match="piperg2p"):
        validate_release_artifacts(tmp_path, "v0.2.0")


def test_rejects_private_ledger_files_in_sdist(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, private_sdist_path=".ledger/taskledger/config.toml")

    with pytest.raises(ValueError, match="private ledger paths"):
        validate_release_artifacts(tmp_path, "v0.2.0")
