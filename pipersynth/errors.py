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
