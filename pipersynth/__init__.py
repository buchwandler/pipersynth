"""Independent synthesis runtime for Piper-compatible ONNX voice models."""

from piperg2p import VoiceConfig

from ._version import __version__
from .asset_manager import CacheInfo, VoiceAssetManager, list_cached_voices, list_voices
from .asset_progress import (
    AssetProgressCallback,
    AssetProgressEvent,
    ConsoleAssetProgress,
)
from .assets import VoiceBundle, VoiceMetadata, load_catalog_voice
from .convenience import synthesize, synthesize_to_wav
from .errors import (
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    CatalogUnavailableError,
    ConfigFileNotFoundError,
    InvalidSpeakerError,
    InvalidSynthesisConfigError,
    ModelFileNotFoundError,
    ModelInferenceError,
    ModelLoadError,
    OfflineAssetError,
    OptionalDependencyError,
    PiperSynthError,
    SessionCreationError,
    SynthesisError,
    TextPreparationError,
    UnsupportedModelError,
    VoiceClosedError,
    VoiceNotFoundError,
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
    "AssetCacheError",
    "AssetDownloadError",
    "AssetError",
    "AssetProgressCallback",
    "AssetProgressEvent",
    "GenerationConfig",
    "CacheInfo",
    "CatalogUnavailableError",
    "ConsoleAssetProgress",
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
    "OfflineAssetError",
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
    "list_cached_voices",
    "TimingDiagnostics",
    "VoiceBundle",
    "VoiceConfig",
    "available_providers",
    "build_pipeline",
    "load_catalog_voice",
    "list_voices",
    "synthesize",
    "synthesize_to_wav",
    "VoiceAssetManager",
    "VoiceMetadata",
    "VoiceNotFoundError",
    "__version__",
]
