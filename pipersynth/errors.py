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


class SynthesisError(PiperSynthError):
    """Speech synthesis failed."""


class InvalidRequestError(SynthesisError, ValueError):
    """A synthesis request violates the public request contract."""


class EmptyTextError(InvalidRequestError):
    """A synthesis request contains no speakable text."""


class InvalidLanguageError(InvalidRequestError):
    """A synthesis request has an empty or model-incompatible language."""


class InvalidLinguisticTokensError(InvalidRequestError):
    """Caller-provided linguistic tokens do not match the source text."""


class InvalidPronunciationError(InvalidRequestError):
    """A pronunciation override is invalid for the source text."""


class SynthesisInputTooLongError(SynthesisError, ValueError):
    """A complete atomic request exceeds a known Piper frontend capacity."""

    def __init__(
        self,
        message: str | None = None,
        *,
        text_length: int | None = None,
        phoneme_count: int | None = None,
        max_phonemes: int | None = None,
        model_id: str | None = None,
    ) -> None:
        self.text_length = text_length
        self.phoneme_count = phoneme_count
        self.max_phonemes = max_phonemes
        self.model_id = model_id
        if message is None:
            details = []
            if text_length is not None:
                details.append(f"text_length={text_length}")
            if phoneme_count is not None:
                details.append(f"phoneme_count={phoneme_count}")
            if max_phonemes is not None:
                details.append(f"max_phonemes={max_phonemes}")
            if model_id is not None:
                details.append(f"model_id={model_id!r}")
            message = "synthesis request exceeds model capacity"
            if details:
                message += " (" + ", ".join(details) + ")"
        super().__init__(message)


class UnsupportedFeatureError(SynthesisError):
    """A requested synthesis capability is not supported by this engine."""


class InvalidSpeakerError(SynthesisError, ValueError):
    """A speaker value is invalid for the loaded voice."""


class InvalidSynthesisConfigError(SynthesisError, ValueError):
    """A synthesis configuration value is invalid."""


class ModelInferenceError(SynthesisError):
    """The model returned unusable output or inference failed."""


class VoiceClosedError(SynthesisError, RuntimeError):
    """An operation was attempted after a voice was closed."""


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
