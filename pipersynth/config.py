from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .errors import InvalidSynthesisConfigError
from .session import ProviderConfig, ProviderSpec


def _validate_number(value: float | None, name: str, minimum: float) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSynthesisConfigError(f"{name} must be a finite number")
    if not np.isfinite(value) or value < minimum:
        raise InvalidSynthesisConfigError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Immutable per-run speech generation policy."""

    speaker: int | str | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    normalize_audio: bool = True
    volume: float = 1.0
    sentence_silence: float = 0.0

    def __post_init__(self) -> None:
        if self.speaker is not None and (
            isinstance(self.speaker, bool) or not isinstance(self.speaker, (int, str))
        ):
            raise InvalidSynthesisConfigError("speaker must be an integer, name, or None")
        _validate_number(self.length_scale, "length_scale", np.finfo(float).tiny)
        _validate_number(self.noise_scale, "noise_scale", 0.0)
        _validate_number(self.noise_w_scale, "noise_w_scale", 0.0)
        if not isinstance(self.normalize_audio, bool):
            raise InvalidSynthesisConfigError("normalize_audio must be a bool")
        _validate_number(self.volume, "volume", 0.0)
        _validate_number(self.sentence_silence, "sentence_silence", 0.0)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable model, frontend, provider, and output configuration."""

    model_path: Path | str
    config_path: Path | str | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    providers: tuple[ProviderSpec | ProviderConfig, ...] | None = None
    provider_options: Mapping[str, Any] | None = None
    session_options: Any | None = None
    frontend_options: Mapping[str, Any] | None = None
    text_preparation: Literal["identity", "spokenform"] = "identity"
    language: str | None = None
    retain_unit_audio: bool = False
    return_diagnostics: bool = True

    def __post_init__(self) -> None:
        if self.text_preparation not in {"identity", "spokenform"}:
            raise ValueError("text_preparation must be 'identity' or 'spokenform'")
        if self.text_preparation == "spokenform" and not self.language:
            raise ValueError("language is required when text_preparation='spokenform'")
        if self.providers is not None:
            object.__setattr__(self, "providers", tuple(self.providers))
        if self.provider_options is not None:
            object.__setattr__(self, "provider_options", dict(self.provider_options))
        if self.frontend_options is not None:
            object.__setattr__(self, "frontend_options", dict(self.frontend_options))
        if not isinstance(self.retain_unit_audio, bool):
            raise ValueError("retain_unit_audio must be a bool")
        if not isinstance(self.return_diagnostics, bool):
            raise ValueError("return_diagnostics must be a bool")
