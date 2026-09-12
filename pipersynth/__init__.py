"""Independent synthesis runtime for Piper-compatible ONNX voice models."""

from piperg2p import VoiceConfig

from ._version import __version__
from .assets import VoiceBundle, load_catalog_voice
from .config import GenerationConfig, PipelineConfig
from .diagnostics import RuntimeDiagnostics, TimingDiagnostics
from .errors import (
    ConfigFileNotFoundError,
    InvalidSpeakerError,
    InvalidSynthesisConfigError,
    ModelFileNotFoundError,
    ModelInferenceError,
    ModelLoadError,
    OptionalDependencyError,
    PiperSynthError,
    SessionCreationError,
    SynthesisError,
    TextPreparationError,
    UnsupportedModelError,
    VoiceClosedError,
)
from .pipeline import PiperPipeline, PreparedAudioUnits, build_pipeline
from .preparation import IdentityTextPreparer, PreparedTextResult, SpokenformTextPreparer
from .session import ProviderConfig, available_providers
from .types import (
    AudioChunk,
    AudioResult,
    AudioUnitDescriptor,
    AudioUnitResult,
    SynthesisConfig,
)
from .voice import PiperVoice

__all__ = [
    "AudioChunk",
    "AudioResult",
    "AudioUnitDescriptor",
    "AudioUnitResult",
    "GenerationConfig",
    "IdentityTextPreparer",
    "PiperPipeline",
    "PiperSynthError",
    "ConfigFileNotFoundError",
    "InvalidSpeakerError",
    "InvalidSynthesisConfigError",
    "ModelFileNotFoundError",
    "ModelInferenceError",
    "ModelLoadError",
    "OptionalDependencyError",
    "PiperVoice",
    "PipelineConfig",
    "PreparedAudioUnits",
    "PreparedTextResult",
    "ProviderConfig",
    "RuntimeDiagnostics",
    "SessionCreationError",
    "SynthesisError",
    "TextPreparationError",
    "UnsupportedModelError",
    "VoiceClosedError",
    "SpokenformTextPreparer",
    "SynthesisConfig",
    "TimingDiagnostics",
    "VoiceBundle",
    "VoiceConfig",
    "available_providers",
    "build_pipeline",
    "load_catalog_voice",
    "__version__",
]
