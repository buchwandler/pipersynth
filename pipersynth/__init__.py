"""Piper speech-synthesis engine for Piper-compatible ONNX voices."""

from piperg2p import VoiceConfig

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.1.1"
    __version_tuple__ = (0, 1, 1)

from .asset_manager import CacheInfo, VoiceAssetManager, list_cached_voices, list_voices
from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress
from .assets import VoiceBundle, VoiceMetadata, load_catalog_voice
from .convenience import synthesize, synthesize_to_wav
from .diagnostics import RuntimeDiagnostics
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
    UnsupportedModelError,
    VoiceClosedError,
    VoiceNotFoundError,
)
from .session import ProviderConfig, ProviderSpec, available_providers
from .types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedChunk,
    RenderedSegment,
    SynthesisConfig,
    SynthesisSegment,
    TextChunkingConfig,
    TextSplitMode,
)
from .voice import PiperVoice
from .voice_level import (
    CalibrationDataError,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelApplication,
    VoiceLevelCalibration,
    VoiceLevelConfig,
    VoiceLevelMode,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibration,
)

__all__ = [
    "AssetCacheError",
    "AssetDownloadError",
    "AssetError",
    "AssetProgressCallback",
    "AssetProgressEvent",
    "CacheInfo",
    "CalibrationDataError",
    "CatalogUnavailableError",
    "ConfigFileNotFoundError",
    "ConsoleAssetProgress",
    "InvalidSpeakerError",
    "InvalidSynthesisConfigError",
    "LinguisticToken",
    "ModelFileNotFoundError",
    "ModelInferenceError",
    "ModelLoadError",
    "OfflineAssetError",
    "OptionalDependencyError",
    "PiperSynthError",
    "PiperVoice",
    "PronunciationOverride",
    "ProviderConfig",
    "ProviderSpec",
    "RenderedChunk",
    "RenderedSegment",
    "RuntimeDiagnostics",
    "SessionCreationError",
    "SynthesisConfig",
    "SynthesisError",
    "SynthesisSegment",
    "TextChunkingConfig",
    "TextSplitMode",
    "UnsupportedModelError",
    "VoiceAssetManager",
    "VoiceBundle",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceClosedError",
    "VoiceConfig",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "VoiceMetadata",
    "VoiceNotFoundError",
    "__version__",
    "__version_tuple__",
    "apply_voice_level_calibration",
    "available_providers",
    "default_voice_calibration",
    "list_cached_voices",
    "list_voices",
    "load_catalog_voice",
    "load_voice_calibration",
    "synthesize",
    "synthesize_to_wav",
]
