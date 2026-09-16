from __future__ import annotations


class PiperSynthError(Exception):
    """Base class for PiperSynth errors."""


class ModelLoadError(PiperSynthError):
    """The voice model or its runtime could not be loaded."""


class ModelFileNotFoundError(ModelLoadError, FileNotFoundError):
    """The requested ONNX model file does not exist."""


class ConfigFileNotFoundError(ModelLoadError, FileNotFoundError):
    """The requested voice configuration file does not exist."""


class UnsupportedModelError(ModelLoadError):
    """The model does not implement the supported Piper contract."""


class SessionCreationError(ModelLoadError):
    """The inference session could not be created."""


class PlanRenderingError(PiperSynthError):
    """An UtterancePlan could not be rendered by the active Piper voice."""


class UnsupportedPlanLanguageError(PlanRenderingError):
    """A plan segment language is unsupported by the active Piper voice."""


class UnsupportedPlanDirectiveError(PlanRenderingError):
    """A plan directive is unsupported by PiperSynth."""


class VoiceBindingError(PlanRenderingError):
    """A logical plan voice cannot be bound to the active Piper voice."""


class PlanSampleRateMismatchError(PlanRenderingError):
    """Rendered voices do not share the output sample rate."""


class SynthesisError(PiperSynthError):
    """Speech synthesis failed."""


class InvalidSpeakerError(SynthesisError, ValueError):
    """A speaker value is invalid for the loaded voice."""


class InvalidSynthesisConfigError(SynthesisError, ValueError):
    """A synthesis configuration value is invalid."""


class ModelInferenceError(SynthesisError):
    """The model returned unusable output or inference failed."""


class VoiceClosedError(SynthesisError, RuntimeError):
    """An operation was attempted after a voice was closed."""


class TextPreparationError(PiperSynthError):
    """Text preparation failed."""


class OptionalDependencyError(PiperSynthError, ImportError):
    """An optional feature was requested without its dependency installed."""


class AssetError(PiperSynthError):
    """Base class for managed voice asset failures."""


class VoiceNotFoundError(AssetError):
    """The requested catalog voice or alias does not exist."""


class AssetDownloadError(AssetError):
    """A managed voice artifact could not be downloaded or verified."""


class AssetCacheError(AssetError):
    """A cached catalog or voice artifact is invalid or unusable."""


class OfflineAssetError(AssetError):
    """A required asset is unavailable while offline mode is enabled."""


class CatalogUnavailableError(AssetError):
    """The voice catalog could not be loaded or refreshed."""
