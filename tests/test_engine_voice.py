from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from piperg2p import PhonemeSentence, VoiceConfig

import pipersynth.voice as voice_module
from pipersynth import (
    InvalidLanguageError,
    InvalidRequestError,
    InvalidSpeakerError,
    LinguisticToken,
    ModelFileNotFoundError,
    ModelInferenceError,
    PiperVoice,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisInputTooLongError,
    SynthesisRequest,
    SynthesisResult,
    VoiceClosedError,
    VoiceLevelConfig,
)


def voice_config(num_speakers: int = 1, *, phoneme_type: str = "text") -> VoiceConfig:
    return VoiceConfig.from_dict(
        {
            "num_symbols": 6,
            "num_speakers": num_speakers,
            "audio": {"sample_rate": 22050},
            "phoneme_type": phoneme_type,
            "espeak": {"voice": "en-us"},
            "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3], "b": [4], " ": [5]},
            "speaker_id_map": {"alice": 1} if num_speakers > 1 else {},
            "default_speaker_id": 2 if num_speakers > 2 else 0,
        }
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], dict[str, Any]]] = []

    def infer(self, ids: list[int], **kwargs: Any) -> Any:
        self.calls.append((tuple(ids), kwargs))
        return SimpleNamespace(
            audio=np.array([[[0.2, -0.4]]], dtype=np.float32),
            sample_rate=22050,
        )


class FakeG2P:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result = SimpleNamespace(
            sentences=(
                PhonemeSentence(("a",), (3,), warnings=("first sentence",)),
                PhonemeSentence((), ()),
                PhonemeSentence(("b",), (4,), warnings=("last sentence",)),
            ),
            warnings=("frontend warning",),
            diagnostics=SimpleNamespace(backend="fake"),
        )

    def phonemize_prepared(self, text: str, **kwargs: Any) -> Any:
        self.calls.append((text, kwargs))
        return self.result


def test_synthesize_passes_exact_request_to_g2p_and_runs_one_inference() -> None:
    runtime = FakeRuntime()
    g2p = FakeG2P()
    factory_calls: list[tuple[str, Any, dict[str, Any]]] = []

    def make_g2p(language: str, *, config: Any, **options: Any) -> FakeG2P:
        factory_calls.append((language, config, options))
        return g2p

    voice = PiperVoice(
        runtime,
        voice_config(3),
        g2p_factory=make_g2p,
        g2p_options={"use_cli": True},
    )
    request = SynthesisRequest(
        id="line-17",
        text="ab! ab?",
        language="en-us",
        speaker="alice",
        pronunciation_overrides=(
            PronunciationOverride(4, 6, phonemes="b", language="de-de", stress="2"),
        ),
        tokens=(
            LinguisticToken(
                0,
                2,
                text="ab",
                pos="NOUN",
                tag="NN",
                lemma="ab",
                language="en-us",
                morph="Number=Sing",
            ),
        ),
    )
    result = voice.synthesize(
        request,
        config=SynthesisConfig(
            length_scale=0.8,
            noise_scale=0.2,
            noise_w_scale=0.3,
            normalize_audio=False,
            output_gain=0.5,
        ),
    )

    assert isinstance(result, SynthesisResult)
    assert result.id == "line-17"
    assert result.text == request.text
    assert result.language == "en-us"
    assert result.audio.tolist() == pytest.approx([0.1, -0.2])
    assert result.audio.dtype == np.float32
    assert result.word_timings == ()
    assert result.supports_timestamps is False
    assert result.metadata["phonemes"] == ("a", "b")
    assert result.metadata["phoneme_ids"] == (3, 4)
    assert result.warnings == ("frontend warning", "first sentence", "last sentence")
    assert g2p.calls[0][0] == request.text
    assert len(g2p.calls) == 1
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] == (3, 4)
    assert runtime.calls[0][1] == {
        "speaker_id": 1,
        "length_scale": 0.8,
        "noise_scale": 0.2,
        "noise_w": 0.3,
    }
    override = g2p.calls[0][1]["overrides"][0]
    assert (override.char_start, override.char_end) == (4, 6)
    assert dict(override.attrs) == {"ph": "b", "lang": "de-de", "stress": 2}
    annotation = g2p.calls[0][1]["annotations"][0]
    assert annotation.morph == "Number=Sing"
    assert annotation.text == "ab"
    assert annotation.pos == "NOUN"
    assert annotation.tag == "NN"
    assert annotation.lemma == "ab"
    assert annotation.language == "en-us"
    assert (annotation.start, annotation.end) == (0, 2)
    assert factory_calls == [("en-us", voice.config, {"use_cli": True})]


def test_synthesize_text_is_strict_and_does_not_import_phrasplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def block_phrasplit(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] == "phrasplit":
            raise AssertionError("strict synthesis must not import Phrasplit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_phrasplit)
    runtime = FakeRuntime()
    g2p = FakeG2P()
    voice = PiperVoice(runtime, voice_config(), g2p_factory=lambda language, *, config: g2p)
    text = "Dr. Smith measured 3.14 volts. The U.S. team left. Final sentence."

    result = voice.synthesize_text(text, language="en-us", id="prepared")

    assert result.text == text
    assert [call[0] for call in g2p.calls] == [text]
    assert [call[0] for call in runtime.calls] == [(3, 4)]


def test_known_capacity_rejects_whole_request_before_inference() -> None:
    runtime = FakeRuntime()
    runtime.max_phonemes = 1
    voice = PiperVoice(
        runtime,
        voice_config(),
        g2p_factory=lambda language, *, config: FakeG2P(),
    )

    with pytest.raises(SynthesisInputTooLongError) as caught:
        voice.synthesize(SynthesisRequest("long", "long text", "en-us"))

    assert caught.value.text_length == len("long text")
    assert caught.value.phoneme_count == 2
    assert caught.value.max_phonemes == 1
    assert runtime.calls == []


def test_unknown_capacity_does_not_invent_a_limit_or_split() -> None:
    runtime = FakeRuntime()
    voice = PiperVoice(
        runtime,
        voice_config(),
        g2p_factory=lambda language, *, config: FakeG2P(),
    )

    result = voice.synthesize_text("A long request. More words.", language="en-us")

    assert result.audio.size == 2
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] == (3, 4)


def test_incompatible_model_language_is_typed() -> None:
    voice = PiperVoice(FakeRuntime(), voice_config(phoneme_type="espeak"))

    with pytest.raises(InvalidLanguageError, match="incompatible"):
        voice.synthesize(SynthesisRequest("id", "hello", "de-de"))


def test_g2p_and_backend_errors_are_translated_to_typed_errors() -> None:
    voice = PiperVoice(
        FakeRuntime(),
        voice_config(),
        g2p_factory=lambda language, *, config: SimpleNamespace(
            phonemize_prepared=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("raw"))
        ),
    )
    with pytest.raises(InvalidRequestError, match="PiperG2P") as g2p_error:
        voice.synthesize(SynthesisRequest("id", "hello", "en-us"))
    assert isinstance(g2p_error.value.__cause__, ValueError)

    class BrokenRuntime(FakeRuntime):
        def infer(self, ids: list[int], **kwargs: Any) -> Any:
            raise RuntimeError("backend details")

    broken = PiperVoice(
        BrokenRuntime(),
        voice_config(),
        g2p_factory=lambda language, *, config: FakeG2P(),
    )
    with pytest.raises(ModelInferenceError, match="inference failed"):
        broken.synthesize(SynthesisRequest("id", "hello", "en-us"))


def test_synthesize_rejects_non_request_values() -> None:
    voice = PiperVoice(FakeRuntime(), voice_config())

    with pytest.raises(InvalidRequestError):
        voice.synthesize("not a request")  # type: ignore[arg-type]


def test_local_voice_opens_through_onnxvoice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "voice.onnx"
    config_path = Path(f"{model}.json")
    model.write_bytes(b"model")
    config_path.write_text(json.dumps(voice_config().to_dict()))
    runtime = FakeRuntime()
    opened: dict[str, Any] = {}

    def open_local(model_path: Path, config_file: Path, **kwargs: Any) -> FakeRuntime:
        opened.update(model_path=model_path, config_path=config_file, options=kwargs)
        return runtime

    monkeypatch.setattr(voice_module, "open_local_voice", open_local)
    g2p = FakeG2P()
    voice = PiperVoice.from_local(
        model,
        g2p_factory=lambda language, *, config, **options: g2p,
        g2p_options={"use_cli": True},
    )

    assert voice.runtime is runtime
    assert voice.model_path == model
    assert voice.config_path == config_path
    assert opened["options"] == {
        "providers": None,
        "provider_options": None,
        "session_options": None,
    }
    result = voice.synthesize_text("Loaded prepared text.", language="en-us")
    assert result.text == "Loaded prepared text."
    assert g2p.calls[0][0] == "Loaded prepared text."
    voice.close()


def test_closed_voice_raises_typed_error() -> None:
    voice = PiperVoice(FakeRuntime(), voice_config())
    voice.close()

    with pytest.raises(VoiceClosedError):
        voice.synthesize_text("hello", language="en-us")


def test_speaker_names_ids_defaults_and_invalid_values() -> None:
    voice = PiperVoice(FakeRuntime(), voice_config(3))
    assert voice.resolve_speaker_id("alice") == 1
    assert voice.resolve_speaker_id(2) == 2
    assert voice.resolve_speaker_id(None) == 2
    with pytest.raises(InvalidSpeakerError, match="unknown speaker"):
        voice.resolve_speaker_id("guest")
    with pytest.raises(InvalidSpeakerError, match="outside"):
        voice.resolve_speaker_id(3)
    with pytest.raises(InvalidSpeakerError, match="unknown speaker"):
        voice.synthesize(SynthesisRequest("id", "hello", "en-us", speaker="guest"))
    with pytest.raises(InvalidSpeakerError, match="outside"):
        voice.synthesize(SynthesisRequest("id", "hello", "en-us", speaker=3))


def test_missing_model_file_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ModelFileNotFoundError):
        PiperVoice.from_local(tmp_path / "missing.onnx")


def test_synthesis_identity_includes_audio_inputs_but_excludes_transient_options() -> None:
    installation = SimpleNamespace(
        id="catalog-voice",
        metadata={"quality": "medium", "source_revision": "revision-7"},
    )

    def render(cache_dir: str, progress: Any, *, length_scale: float = 0.8) -> SynthesisResult:
        runtime = FakeRuntime()
        g2p = FakeG2P()
        voice = PiperVoice(
            runtime,
            voice_config(),
            installation=installation,
            g2p_factory=lambda language, *, config, **options: g2p,
            g2p_options={
                "use_cli": True,
                "cache_dir": cache_dir,
                "progress": progress,
                "provider_options": {"cache_dir": cache_dir, "dialect": "en-us"},
            },
        )
        return voice.synthesize(
            SynthesisRequest("id", "hello", "en-us"),
            config=SynthesisConfig(
                length_scale=length_scale, normalize_audio=False, output_gain=0.5
            ),
        )

    first = render("/tmp/cache-one", lambda *_: None)
    second = render("/different/cache", lambda *_: None)
    identity = first.metadata["synthesis_identity"]
    assert identity["pipersynth_version"]
    assert identity["model_id"] == "catalog-voice"
    assert identity["model_revision"] == "revision-7"
    assert identity["quality"] == "medium"
    assert identity["speaker_id"] == 0
    assert identity["language"] == "en-us"
    assert identity["length_scale"] == 0.8
    assert identity["normalize_audio"] is False
    assert identity["output_gain"] == 0.5
    assert identity["frontend"]["g2p_version"]
    assert identity["frontend"]["options"] == {
        "provider_options": {"dialect": "en-us"},
        "use_cli": True,
    }
    assert first.metadata["synthesis_hash"] == second.metadata["synthesis_hash"]
    assert first.metadata["voice_level"] == {
        "mode": "off",
        "applied": False,
        "gain_db": 0.0,
        "source": "off",
        "calibration_key": "piper:catalog-voice:medium:speaker-0",
        "reason": "voice-level calibration is disabled",
        "catalog_revision": None,
    }
    changed = render("/tmp/cache-one", lambda *_: None, length_scale=0.9)
    assert changed.metadata["synthesis_hash"] != first.metadata["synthesis_hash"]


def test_calibration_missing_states_and_override_are_observable() -> None:
    def render(installation: Any, config: VoiceLevelConfig) -> dict[str, Any]:
        voice = PiperVoice(
            FakeRuntime(),
            voice_config(),
            installation=installation,
            g2p_factory=lambda language, *, config: FakeG2P(),
        )
        return voice.synthesize_text(
            "hello",
            language="en-us",
            config=SynthesisConfig(normalize_audio=False, voice_level=config),
        ).metadata["voice_level"]

    missing_identity = render(None, VoiceLevelConfig(mode="calibrated"))
    assert missing_identity["source"] == "missing_identity"
    assert missing_identity["reason"] == "the voice has no stable calibration identity"

    unmeasured = render(
        SimpleNamespace(id="unmeasured", metadata={"quality": "medium"}),
        VoiceLevelConfig(mode="calibrated"),
    )
    assert unmeasured["source"] == "missing_calibration"
    assert len(unmeasured["catalog_revision"]) == 64

    override = render(
        None,
        VoiceLevelConfig(mode="calibrated", gain_db=0.0),
    )
    assert override["source"] == "override"
    assert override["applied"] is False
    assert override["reason"] == "an explicit gain_db override was selected"
