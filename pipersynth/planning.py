from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from piperg2p import VoiceConfig
from ttsplan import (
    LinguisticsConfig,
    PauseConfig,
    PlannerConfig,
    SSMDConfig,
    normalize_language,
)
from ttsplan.exceptions import ConfigurationError

from .config import PipelineConfig
from .errors import ConfigFileNotFoundError


@dataclass(frozen=True, slots=True)
class ResolvedVoiceMetadata:
    """Voice sidecar data needed before constructing a Piper runtime."""

    config_path: Path
    voice_config: VoiceConfig
    planner_language: str


def inspect_voice_config(config: PipelineConfig) -> ResolvedVoiceMetadata:
    """Read local Piper metadata without creating an ONNX session."""

    config_path = Path(config.config_path) if config.config_path is not None else Path(f"{config.model_path}.json")
    if not config_path.exists():
        raise ConfigFileNotFoundError(f"Voice config file does not exist: {config_path}")
    voice_config = VoiceConfig.from_json(config_path)
    language = getattr(voice_config, "espeak_voice", "")
    if not isinstance(language, str) or not language.strip():
        raise ConfigurationError(
            "A planner language could not be resolved. Set PipelineConfig.language or provide voice metadata."
        )
    return ResolvedVoiceMetadata(config_path, voice_config, normalize_language(language))


def planner_config_from_pipersynth(
    config: PipelineConfig,
    *,
    language: str | None = None,
    unit: Literal["paragraph", "sentence"] | None = None,
) -> PlannerConfig:
    """Map PiperSynth's public planner settings to TTSPlan configuration."""

    resolved_language = language or config.language
    if not resolved_language:
        raise ConfigurationError("A planner language could not be resolved")
    pauses = config.pauses
    if config.generation.sentence_silence:
        pauses = replace(pauses, sentence=config.generation.sentence_silence)
    return PlannerConfig(
        language=normalize_language(resolved_language, dict(config.language_aliases)),
        document_format=config.document_format,
        text_preparation=config.text_preparation,
        unit=unit or config.unit,
        pauses=pauses,
        linguistics=config.linguistics,
        ssmd=config.ssmd,
        overlap_mode=config.overlap_mode,
        language_aliases=dict(config.language_aliases),
        diagnostics=config.planner_diagnostics,
    )


__all__ = [
    "LinguisticsConfig",
    "PauseConfig",
    "PlannerConfig",
    "ResolvedVoiceMetadata",
    "SSMDConfig",
    "inspect_voice_config",
    "normalize_language",
    "planner_config_from_pipersynth",
]
