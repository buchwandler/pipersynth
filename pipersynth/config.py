from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from utterplan import LinguisticsConfig, PauseConfig, SSMDConfig

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
    is_phonemes: bool = False
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
        if not isinstance(self.is_phonemes, bool):
            raise InvalidSynthesisConfigError("is_phonemes must be a bool")
        if self.sentence_silence:
            warnings.warn(
                "sentence_silence is deprecated; configure semantic pauses through UtterPlan",
                DeprecationWarning,
                stacklevel=2,
            )
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

    language: str | None = None
    document_format: Literal["plain", "ssmd"] = "plain"
    text_preparation: Literal["identity", "spokenform"] = "identity"
    unit: Literal["paragraph", "sentence"] = "paragraph"
    pauses: PauseConfig = field(default_factory=PauseConfig)
    linguistics: LinguisticsConfig = field(default_factory=LinguisticsConfig)
    ssmd: SSMDConfig = field(default_factory=SSMDConfig)
    overlap_mode: Literal["snap", "strict"] = "snap"
    language_aliases: Mapping[str, str] = field(default_factory=dict)
    planner_diagnostics: bool = True
    directive_policy: Literal["error", "warn", "ignore"] = "error"
    language_policy: Literal["strict", "allow"] = "strict"

    retain_unit_audio: bool = False
    return_diagnostics: bool = True

    def __post_init__(self) -> None:
        if self.document_format not in {"plain", "ssmd"}:
            raise ValueError("document_format must be 'plain' or 'ssmd'")
        if self.text_preparation not in {"identity", "spokenform"}:
            raise ValueError("text_preparation must be 'identity' or 'spokenform'")
        if self.unit not in {"paragraph", "sentence"}:
            raise ValueError("unit must be 'paragraph' or 'sentence'")
        if self.overlap_mode not in {"snap", "strict"}:
            raise ValueError("overlap_mode must be 'snap' or 'strict'")
        if self.directive_policy not in {"error", "warn", "ignore"}:
            raise ValueError("directive_policy must be 'error', 'warn', or 'ignore'")
        if self.language_policy not in {"strict", "allow"}:
            raise ValueError("language_policy must be 'strict' or 'allow'")
        if not isinstance(self.planner_diagnostics, bool):
            raise ValueError("planner_diagnostics must be a bool")
        if not isinstance(self.language_aliases, Mapping):
            raise ValueError("language_aliases must be a string-to-string mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.language_aliases.items()
        ):
            raise ValueError("language_aliases must contain only string keys and values")
        if self.providers is not None:
            object.__setattr__(self, "providers", tuple(self.providers))
        if self.provider_options is not None:
            object.__setattr__(self, "provider_options", dict(self.provider_options))
        if self.frontend_options is not None:
            object.__setattr__(self, "frontend_options", dict(self.frontend_options))
        object.__setattr__(self, "language_aliases", dict(self.language_aliases))
        if not isinstance(self.retain_unit_audio, bool):
            raise ValueError("retain_unit_audio must be a bool")
        if not isinstance(self.return_diagnostics, bool):
            raise ValueError("return_diagnostics must be a bool")
