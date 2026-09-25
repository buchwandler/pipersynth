from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._onnxvoice import available_providers as onnxvoice_available_providers
from .errors import OptionalDependencyError

ProviderSpec = str | tuple[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """One ONNX Runtime execution provider and its options."""

    name: str
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider name must not be empty")
        object.__setattr__(self, "options", dict(self.options or {}))


def available_providers() -> tuple[str, ...]:
    """Return providers exposed by OnnxVoice's optional runtime."""
    try:
        return onnxvoice_available_providers()
    except OptionalDependencyError as exc:
        raise OptionalDependencyError(
            "ONNX Runtime is required for acoustic inference. Install pipersynth[cpu] "
            "or pipersynth[gpu]."
        ) from exc


__all__ = ["ProviderConfig", "ProviderSpec", "available_providers"]
