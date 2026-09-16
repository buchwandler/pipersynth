from __future__ import annotations

from pathlib import Path

import pytest

from examples._output import artefact_path


def test_artefact_path_uses_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIPERSYNTH_EXAMPLE_OUTPUT_DIR", str(tmp_path))

    path = artefact_path("nested/output.wav")

    assert path == (tmp_path / "nested/output.wav").resolve()
    assert path.parent.is_dir()


def test_artefact_path_rejects_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPERSYNTH_EXAMPLE_OUTPUT_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="escapes"):
        artefact_path("../outside.wav")
