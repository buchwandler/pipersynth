"""Independent synthesis runtime for Piper-compatible ONNX voice models."""

from piperg2p import VoiceConfig
from ttsplan import LinguisticsConfig, PauseConfig, SSMDConfig, TTSPlan, TTSPlanner

from ._version import __version__
from .asset_manager import CacheInfo, VoiceAssetManager, list_cached_voices, list_voices
from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress
from .assets import VoiceBundle, VoiceMetadata, load_catalog_voice
from .config import GenerationConfig, PipelineConfig
from .convenience import synthesize, synthesize_to_wav
from .diagnostics import RuntimeDiagnostics, TimingDiagnostics
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
    PlanRenderingError,
    PlanSampleRateMismatchError,
    SessionCreationError,
    SynthesisError,
    TextPreparationError,
    UnsupportedModelError,
    UnsupportedPlanDirectiveError,
    UnsupportedPlanLanguageError,
    VoiceBindingError,
    VoiceClosedError,
    VoiceNotFoundError,
)
from .pipeline import PiperPipeline, PreparedAudioUnits, build_pipeline
from .preparation import IdentityTextPreparer, PreparedTextResult, SpokenformTextPreparer
from .session import ProviderConfig, available_providers
from .types import AudioChunk, AudioResult, AudioUnitDescriptor, AudioUnitResult, SynthesisConfig
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
    "LinguisticsConfig",
    "PauseConfig",
    "SSMDConfig",
    "TTSPlan",
    "TTSPlanner",
    "CacheInfo",
    "VoiceAssetManager",
    "CatalogUnavailableError",
    "ConsoleAssetProgress",
    "IdentityTextPreparer",
    "PiperPipeline",
    "PiperSynthError",
    "PlanRenderingError",
    "PlanSampleRateMismatchError",
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
    "TimingDiagnostics",
    "UnsupportedModelError",
    "UnsupportedPlanDirectiveError",
    "UnsupportedPlanLanguageError",
    "VoiceBindingError",
    "VoiceClosedError",
    "SpokenformTextPreparer",
    "SynthesisConfig",
    "VoiceBundle",
    "VoiceConfig",
    "VoiceMetadata",
    "VoiceNotFoundError",
    "available_providers",
    "build_pipeline",
    "list_cached_voices",
    "load_catalog_voice",
    "list_voices",
    "synthesize",
    "synthesize_to_wav",
    "__version__",
]
