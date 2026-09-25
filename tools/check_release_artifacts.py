from __future__ import annotations

import os
import re
import tarfile
import zipfile
from email import policy
from email.parser import Parser
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.version import Version

CORE_MINIMUMS = {
    "numpy": "1.23",
    "piperg2p": "0.1.7",
    "onnxvoice": "0.1.10",
    "audiosig": "0.1.4",
}
FORBIDDEN_DEPENDENCIES = {"utterplan", "audiocompose", "ssmd", "phrasplit"}
PRIVATE_SDIST_DIRECTORIES = {".ledger", ".taskledger", ".repairledger", ".releaseledger"}
REQUIRED_WHEEL_FILES = {
    "pipersynth/py.typed",
    "pipersynth/data/voice_level_calibration.json",
}


def _has_minimum(requirements: list[Requirement], name: str, minimum: str) -> bool:
    minimum_version = Version(minimum)
    return any(
        requirement.name == name
        and any(
            spec.operator in (">=", ">", "==") and Version(spec.version) >= minimum_version
            for spec in requirement.specifier
        )
        for requirement in requirements
    )


def _metadata_version(metadata: str) -> str:
    return Parser(policy=policy.default).parsestr(metadata)["Version"]


def validate_release_artifacts(dist: Path, release_tag: str) -> None:
    tag_version = Version(release_tag.removeprefix("v"))
    artifacts = list(dist.iterdir())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one wheel, found {wheels}")
    if len(sdists) != 1:
        raise ValueError(f"expected exactly one sdist, found {sdists}")

    wheel_match = re.fullmatch(r"pipersynth-([^-]+)-.+\.whl", wheels[0].name)
    sdist_match = re.fullmatch(r"pipersynth-([^-]+)\.tar\.gz", sdists[0].name)
    if wheel_match is None or sdist_match is None:
        raise ValueError("distribution filenames do not match the PiperSynth naming convention")
    if Version(wheel_match.group(1)) != tag_version:
        raise ValueError(f"wheel version {wheel_match.group(1)} does not match tag {tag_version}")
    if Version(sdist_match.group(1)) != tag_version:
        raise ValueError(f"sdist version {sdist_match.group(1)} does not match tag {tag_version}")

    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith("METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"expected one wheel METADATA file, found {metadata_names}")
        metadata = Parser(policy=policy.default).parsestr(
            archive.read(metadata_names[0]).decode("utf-8")
        )
        if Version(metadata["Version"]) != tag_version:
            raise ValueError(
                f"wheel metadata version {metadata['Version']} does not match tag {tag_version}"
            )
        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        missing = [
            name
            for name, minimum in CORE_MINIMUMS.items()
            if not _has_minimum(requirements, name, minimum)
        ]
        if missing:
            raise ValueError(f"wheel is missing required dependency floors: {missing}")
        forbidden = sorted(
            requirement.name
            for requirement in requirements
            if requirement.name in FORBIDDEN_DEPENDENCIES
        )
        if forbidden:
            raise ValueError(f"wheel requires removed or unused dependencies: {forbidden}")
        missing_files = sorted(
            path for path in REQUIRED_WHEEL_FILES if not any(name.endswith(path) for name in names)
        )
        if missing_files:
            raise ValueError(f"wheel is missing package data: {missing_files}")

    with tarfile.open(sdists[0], "r:gz") as archive:
        members = archive.getmembers()
        private_paths = [
            member.name
            for member in members
            if PRIVATE_SDIST_DIRECTORIES.intersection(PurePosixPath(member.name).parts)
        ]
        if private_paths:
            raise ValueError(f"sdist contains private ledger paths: {private_paths[:5]}")
        pkg_info_members = [member for member in members if member.name.endswith("PKG-INFO")]
        if not pkg_info_members:
            raise ValueError("sdist does not contain PKG-INFO metadata")
        for member in pkg_info_members:
            pkg_info = archive.extractfile(member)
            if (
                pkg_info is None
                or Version(_metadata_version(pkg_info.read().decode("utf-8"))) != tag_version
            ):
                raise ValueError(f"sdist metadata version does not match tag {tag_version}")


def main() -> None:
    validate_release_artifacts(Path("dist"), os.environ["RELEASE_TAG"])
    print(f"Validated PiperSynth release artifacts for {os.environ['RELEASE_TAG']}")


if __name__ == "__main__":
    main()
