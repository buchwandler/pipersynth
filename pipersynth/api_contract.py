"""Dependency-light declaration of the public request API contract."""

from __future__ import annotations

from dataclasses import dataclass

REQUEST_API_VERSION = 1


@dataclass(frozen=True, slots=True)
class RequestApiContract:
    """Stable capability declaration for request-oriented integrations."""

    version: int
    request_type: str
    result_type: str
    entrypoint: str
    supports_linguistic_tokens: bool
    supports_pronunciation_overrides: bool
    supports_whole_request_phonemes: bool
    supports_named_voice_bundles: bool
    supports_speakers: bool
    supports_word_timings: bool
    supports_voice_level: bool
    caller_owns_text_boundaries: bool


def request_api_contract() -> RequestApiContract:
    """Return the stable request API capabilities without initializing a voice runtime."""
    return RequestApiContract(
        version=REQUEST_API_VERSION,
        request_type="SynthesisRequest",
        result_type="SynthesisResult",
        entrypoint="PiperVoice.synthesize",
        supports_linguistic_tokens=True,
        supports_pronunciation_overrides=True,
        supports_whole_request_phonemes=False,
        supports_named_voice_bundles=True,
        supports_speakers=True,
        supports_word_timings=False,
        supports_voice_level=True,
        caller_owns_text_boundaries=True,
    )
