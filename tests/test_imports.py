from __future__ import annotations

import subprocess
import sys


def test_removed_document_dependencies_and_modules_are_not_required() -> None:
    command = """\
import importlib.abc
import importlib.util
import sys

blocked = {"utterplan", "audiocompose", "ssmd"}

class BlockRemovedDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockRemovedDependencies())
import pipersynth
assert not blocked.intersection(sys.modules)
for module in (
    "pipersynth.audio_job",
    "pipersynth.composition",
    "pipersynth.pipeline",
    "pipersynth.plan_adapter",
    "pipersynth.planning",
    "pipersynth.preparation",
):
    assert importlib.util.find_spec(module) is None
assert not hasattr(pipersynth, "PiperPipeline")
"""
    result = subprocess.run(
        [sys.executable, "-c", command], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
