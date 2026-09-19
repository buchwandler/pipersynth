"""Independent synthesis runtime for Piper-compatible ONNX voice models."""

from piperg2p import VoiceConfig
from utterplan import (
    LinguisticsConfig,
    PauseConfig,
    PlannerConfig,
    SSMDConfig,
    UtterancePlan,
    UtterancePlanner,
)

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.1.1"
    __version_tuple__ = (0, 1, 1)
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
from .loudness_config import LoudnessConfig, PeakPolicy, VoiceLevelingMode, coerce_loudness
from .pipeline import PiperPipeline, PreparedAudioUnits, build_pipeline
from .preparation import IdentityTextPreparer, PreparedTextResult, SpokenformTextPreparer
from .session import ProviderConfig, available_providers
from .types import AudioChunk, AudioResult, AudioUnitDescriptor, AudioUnitResult, SynthesisConfig
from .voice import PiperVoice
from .voice_level import (
    CalibrationDataError,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelApplication,
    VoiceLevelCalibration,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibration,
)

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
    "LoudnessConfig",
    "PeakPolicy",
    "VoiceLevelingMode",
    "coerce_loudness",
    "PauseConfig",
    "PlannerConfig",
    "SSMDConfig",
    "UtterancePlan",
    "UtterancePlanner",
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
    "CalibrationDataError",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "load_voice_calibration",
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
    "__version_tuple__",
]
