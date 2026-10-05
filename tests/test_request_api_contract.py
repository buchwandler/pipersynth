"""Regression tests for the dependency-light public request API contract."""

from __future__ import annotations

import subprocess
import sys

import pipersynth
from pipersynth.api_contract import REQUEST_API_VERSION, RequestApiContract, request_api_contract


def test_request_api_version_is_public_and_one() -> None:
    assert REQUEST_API_VERSION == 1
    assert pipersynth.REQUEST_API_VERSION == 1
    assert pipersynth.RequestApiContract is RequestApiContract
    assert pipersynth.request_api_contract is request_api_contract


def test_request_api_contract_declares_stable_capabilities() -> None:
    assert request_api_contract() == RequestApiContract(
        version=1,
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


def test_contract_probe_resolves_request_api_without_runtime_session() -> None:
    code = r"""
import importlib.abc
import sys

class BlockOnnxRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnxruntime" or fullname.startswith("onnxruntime."):
            raise AssertionError(f"request API inspection imported {fullname}")
        return None

sys.meta_path.insert(0, BlockOnnxRuntime())
import pipersynth

contract = pipersynth.request_api_contract()
assert contract.request_type == "SynthesisRequest"
assert contract.result_type == "SynthesisResult"
assert callable(pipersynth.PiperVoice.synthesize)
assert callable(pipersynth.VoiceAssetManager)
assert not any(name == "onnxruntime" or name.startswith("onnxruntime.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_readio_required_request_api_symbols_and_tokens_are_public() -> None:
    required = (
        "PiperVoice",
        "VoiceAssetManager",
        "SynthesisRequest",
        "SynthesisResult",
        "SynthesisConfig",
        "LinguisticToken",
        "PronunciationOverride",
        "VoiceLevelConfig",
        "SynthesisInputTooLongError",
    )
    assert all(getattr(pipersynth, name, None) is not None for name in required)

    token = pipersynth.LinguisticToken(0, 5, text="Hello")
    request = pipersynth.SynthesisRequest("api-smoke", "Hello", "en-us", tokens=(token,))
    config = pipersynth.SynthesisConfig()
    assert request.tokens == (token,)
    assert request.tokens[0].text == "Hello"
    assert config.voice_level.mode == "off"
    assert callable(pipersynth.PiperVoice.synthesize)
    assert callable(pipersynth.VoiceAssetManager)
