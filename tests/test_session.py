from __future__ import annotations

from typing import Any

import pytest

import pipersynth.session as session
from pipersynth import ProviderConfig
from pipersynth.errors import OptionalDependencyError


def test_provider_config_copies_options() -> None:
    options: dict[str, Any] = {"device_id": 0}
    provider = ProviderConfig("CUDAExecutionProvider", options)
    options["device_id"] = 1
    assert provider.name == "CUDAExecutionProvider"
    assert provider.options == {"device_id": 0}


def test_provider_config_rejects_empty_names() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ProviderConfig("")


def test_available_providers_delegates_to_onnxvoice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "onnxvoice_available_providers", lambda: ("CPUExecutionProvider",))
    assert session.available_providers() == ("CPUExecutionProvider",)


def test_available_providers_gives_an_install_hint_for_missing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> tuple[str, ...]:
        raise OptionalDependencyError("runtime missing")

    monkeypatch.setattr(session, "onnxvoice_available_providers", unavailable)
    with pytest.raises(OptionalDependencyError, match=r"pipersynth\[cpu\].*pipersynth\[gpu\]"):
        session.available_providers()
