from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from piperg2p import PhonemeSentence, VoiceConfig

import pipersynth.voice as voice_module
from pipersynth import PiperVoice
from pipersynth.errors import InvalidSynthesisConfigError
from pipersynth.types import (
    LinguisticToken,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisSegment,
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


class FakeFrontend:
    diagnostics = None

    def close(self) -> None:
        pass


def test_synthesize_forwards_offsets_and_infers_each_nonempty_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    fake_g2p = FakeG2P()
    monkeypatch.setattr(voice_module, "get_g2p", lambda language, *, config: fake_g2p)
    voice = PiperVoice(runtime, voice_config(3), FakeFrontend())  # type: ignore[arg-type]
    segment = SynthesisSegment(
        id="line-17",
        text="ab ab",
        language="en-us",
        speaker="alice",
        pronunciation_overrides=(
            PronunciationOverride(3, 5, phonemes="b", language="de-de", stress="2"),
        ),
        annotations=(
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
        segment,
        config=SynthesisConfig(
            length_scale=0.8,
            noise_scale=0.2,
            noise_w_scale=0.3,
            normalize_audio=False,
            output_gain=0.5,
        ),
    )

    assert result.id == "line-17"
    assert result.audio.tolist() == pytest.approx([0.1, -0.2, 0.1, -0.2])
    assert result.phonemes == ("a", "b")
    assert result.phoneme_ids == (3, 4)
    assert result.warnings == ("frontend warning", "first sentence", "last sentence")
    assert [chunk.segment_id for chunk in result.chunks] == ["line-17", "line-17"]
    assert [chunk.index for chunk in result.chunks] == [0, 1]
    assert [call[0] for call in runtime.calls] == [(3,), (4,)]
    assert all(call[1]["speaker_id"] == 1 for call in runtime.calls)
    assert all(call[1]["length_scale"] == 0.8 for call in runtime.calls)
    assert all(call[1]["noise_scale"] == 0.2 for call in runtime.calls)
    assert all(call[1]["noise_w"] == 0.3 for call in runtime.calls)

    overrides = fake_g2p.calls[0][1]["overrides"]
    assert overrides[0].char_start == 3
    assert overrides[0].char_end == 5
    assert dict(overrides[0].attrs) == {"ph": "b", "lang": "de-de", "stress": 2}
    annotations = fake_g2p.calls[0][1]["annotations"]
    assert annotations[0].morph == "Number=Sing"
    assert annotations[0].text == "ab"
    assert (annotations[0].start, annotations[0].end) == (0, 2)
    assert annotations[0].pos == "NOUN"
    assert annotations[0].tag == "NN"
    assert annotations[0].lemma == "ab"
    assert annotations[0].language == "en-us"
    assert result.diagnostics is not None


def test_iter_chunks_yields_only_inferable_groups() -> None:
    runtime = FakeRuntime()
    voice = PiperVoice(runtime, voice_config(), FakeFrontend())  # type: ignore[arg-type]
    sentence_result = SimpleNamespace(
        sentences=(PhonemeSentence((), ()), PhonemeSentence(("a",), (3,))),
        warnings=(),
        diagnostics=None,
    )
    voice._phonemize_segment = lambda segment: sentence_result  # type: ignore[method-assign]
    segment = SynthesisSegment("id", "a", "en-us")

    chunks = tuple(voice.iter_chunks(segment))

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].segment_id == "id"
    assert len(runtime.calls) == 1


def test_language_must_match_active_espeak_voice_without_span_routing() -> None:
    voice = PiperVoice(FakeRuntime(), voice_config(phoneme_type="espeak"), FakeFrontend())  # type: ignore[arg-type]
    segment = SynthesisSegment("id", "hello", "de-de")
    with pytest.raises(InvalidSynthesisConfigError, match="incompatible"):
        voice.synthesize(segment)


def test_synthesize_ids_keeps_low_level_inference_controls() -> None:
    runtime = FakeRuntime()
    voice = PiperVoice(runtime, voice_config(3), FakeFrontend())  # type: ignore[arg-type]
    audio = voice.synthesize_ids(
        [3, 4],
        SynthesisConfig(
            length_scale=0.75,
            noise_scale=0.4,
            noise_w_scale=0.6,
            normalize_audio=False,
            output_gain=0.5,
        ),
        speaker=2,
    )
    assert audio.tolist() == pytest.approx([0.1, -0.2])
    assert runtime.calls[0][1]["speaker_id"] == 2
    assert runtime.calls[0][1]["length_scale"] == 0.75
    assert runtime.calls[0][1]["noise_scale"] == 0.4
    assert runtime.calls[0][1]["noise_w"] == 0.6


def test_synthesize_text_passes_prepared_text_and_requested_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    frontend = FakeG2P()
    monkeypatch.setattr(voice_module, "get_g2p", lambda language, *, config: frontend)
    voice = PiperVoice(runtime, voice_config(3), FakeFrontend())  # type: ignore[arg-type]

    result = voice.synthesize_text(
        "Already prepared.", language="en-us", id="prepared-1", speaker="alice"
    )

    assert result.id == "prepared-1"
    assert result.text == "Already prepared."
    assert frontend.calls[0][0] == "Already prepared."
    assert runtime.calls[0][1]["speaker_id"] == 1


def test_local_voice_opens_through_onnxvoice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "voice.onnx"
    config_path = Path(f"{model}.json")
    model.write_bytes(b"model")
    config_path.write_text(json.dumps(voice_config().to_dict()))
    runtime = FakeRuntime()
    frontend = FakeFrontend()
    opened: dict[str, Any] = {}

    def open_local(model_path, config_file, **kwargs):
        opened.update(model_path=model_path, config_path=config_file, options=kwargs)
        return runtime

    monkeypatch.setattr(voice_module, "open_local_voice", open_local)
    monkeypatch.setattr(voice_module, "PiperFrontend", lambda config, **kwargs: frontend)

    voice = PiperVoice.from_local(model)

    assert voice.runtime is runtime
    assert voice.model_path == model
    assert voice.config_path == config_path
    assert opened == {
        "model_path": model,
        "config_path": config_path,
        "options": {"providers": None, "provider_options": None, "session_options": None},
    }
    voice.close()


def test_runtime_summaries_are_kept_as_engine_diagnostics() -> None:
    class Runtime(FakeRuntime):
        def infer(self, ids: list[int], **kwargs: Any) -> Any:
            self.calls.append((tuple(ids), kwargs))
            return SimpleNamespace(
                audio=np.array([0.2, -0.4], dtype=np.float32),
                sample_rate=22050,
                timings=np.zeros((1, 2), dtype=np.float32),
                outputs={"duration": np.zeros((1, 2), dtype=np.float32)},
            )

    voice = PiperVoice(Runtime(), voice_config(), FakeFrontend())  # type: ignore[arg-type]
    voice._phonemize_segment = lambda segment: SimpleNamespace(
        sentences=(PhonemeSentence(("a",), (3,)),), warnings=(), diagnostics=None
    )  # type: ignore[method-assign]

    result = voice.synthesize(SynthesisSegment("id", "a", "en-us"))

    assert result.chunks[0].metadata["inference_timing"] == {"shape": [1, 2], "dtype": "float32"}
    assert result.chunks[0].metadata["inference_output"] == {
        "duration": {"shape": [1, 2], "dtype": "float32"}
    }


def test_speaker_names_ids_and_default_are_resolved_within_active_voice() -> None:
    voice = PiperVoice(FakeRuntime(), voice_config(3), FakeFrontend())  # type: ignore[arg-type]
    assert voice.resolve_speaker_id("alice") == 1
    assert voice.resolve_speaker_id(2) == 2
    assert voice.resolve_speaker_id(None) == 2


def test_invalid_speakers_are_rejected() -> None:
    from pipersynth.errors import InvalidSpeakerError

    voice = PiperVoice(FakeRuntime(), voice_config(3), FakeFrontend())  # type: ignore[arg-type]
    with pytest.raises(InvalidSpeakerError, match="unknown speaker"):
        voice.resolve_speaker_id("guest")
    with pytest.raises(InvalidSpeakerError, match="outside"):
        voice.resolve_speaker_id(3)
