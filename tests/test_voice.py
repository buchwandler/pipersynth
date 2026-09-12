import numpy as np
import pytest
from piperg2p import PiperFrontend, VoiceConfig

from pipersynth import PiperVoice, SynthesisConfig


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, outputs, args):
        self.calls.append((outputs, args))
        return [np.asarray([[[0.0, 0.25, -0.5]]], dtype=np.float32)]


def config(num_speakers=1):
    return VoiceConfig.from_dict(
        {
            "num_symbols": 6,
            "num_speakers": num_speakers,
            "audio": {"sample_rate": 22050},
            "phoneme_type": "text",
            "espeak": {"voice": "en-us"},
            "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                "a": [3],
                "b": [4],
                " ": [5],
            },
            "default_speaker_id": 2 if num_speakers > 2 else 0,
        }
    )


def test_onnx_contract_and_normalization():
    cfg = config()
    session = FakeSession()
    voice = PiperVoice(session, cfg, PiperFrontend(cfg))
    audio = voice.synthesize_ids([1, 0, 3, 0, 2])
    _, args = session.calls[0]
    assert args["input"].dtype == np.int64
    assert args["input_lengths"].tolist() == [5]
    assert args["scales"].dtype == np.float32
    assert "sid" not in args
    assert np.isclose(np.max(np.abs(audio)), 1.0)


def test_multispeaker_sid():
    cfg = config(3)
    session = FakeSession()
    voice = PiperVoice(session, cfg, PiperFrontend(cfg))
    voice.synthesize_ids([1, 0, 2], SynthesisConfig(speaker_id=1, normalize_audio=False))
    assert session.calls[0][1]["sid"].tolist() == [1]


def test_multispeaker_default_sid():
    cfg = config(3)
    session = FakeSession()
    voice = PiperVoice(session, cfg, PiperFrontend(cfg))
    voice.synthesize_ids([1, 0, 2], SynthesisConfig(normalize_audio=False))
    assert session.calls[0][1]["sid"].tolist() == [2]


def test_invalid_speaker_rejected():
    cfg = config(3)
    voice = PiperVoice(FakeSession(), cfg, PiperFrontend(cfg))
    with pytest.raises(ValueError):
        voice.synthesize_ids([1, 0, 2], SynthesisConfig(speaker_id=3))


def test_empty_ids_short_circuit():
    cfg = config()
    session = FakeSession()
    voice = PiperVoice(session, cfg, PiperFrontend(cfg))
    audio = voice.synthesize_ids([])
    assert audio.dtype == np.float32
    assert audio.size == 0
    assert not session.calls


class ShapeSession(FakeSession):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def run(self, outputs, args):
        self.calls.append((outputs, args))
        return [np.asarray(self.output)]


def test_output_shape_reduction_and_ambiguous_shape_rejection():
    cfg = config()
    reduced = PiperVoice(ShapeSession([[0.0, 0.5]]), cfg, PiperFrontend(cfg))
    assert reduced.synthesize_ids([1, 2]).shape == (2,)

    ambiguous = PiperVoice(ShapeSession([[0.0, 0.5], [0.1, 0.2]]), cfg, PiperFrontend(cfg))
    with pytest.raises(Exception, match="ambiguous waveform shape"):
        ambiguous.synthesize_ids([1, 2])


def test_speaker_name_and_single_speaker_validation():
    cfg = VoiceConfig.from_dict({**config(3).to_dict(), "speaker_id_map": {"alice": 1}})
    session = FakeSession()
    voice = PiperVoice(session, cfg, PiperFrontend(cfg))
    voice.synthesize_ids([1, 2], SynthesisConfig(speaker_id=voice.resolve_speaker_id("alice")))
    assert session.calls[0][1]["sid"].tolist() == [1]

    single = PiperVoice(FakeSession(), config(), PiperFrontend(config()))
    with pytest.raises(ValueError, match="single-speaker"):
        single.synthesize_ids([1, 2], SynthesisConfig(speaker_id=1))


def test_frontend_warnings_are_propagated():
    cfg = config()
    frontend = PiperFrontend(cfg)
    voice = PiperVoice(FakeSession(), cfg, frontend)
    chunks = list(voice.synthesize("hello"))
    assert chunks
    assert isinstance(chunks[0].warnings, tuple)


def test_close_is_idempotent_and_use_after_close_is_rejected():
    voice = PiperVoice(FakeSession(), config(), PiperFrontend(config()))
    voice.close()
    voice.close()
    with pytest.raises(RuntimeError, match="closed"):
        voice.synthesize_ids([1])


def test_load_uses_default_config_path(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text(__import__("json").dumps(config().to_dict()))
    voice = PiperVoice.load(model, session_factory=lambda path, **kwargs: FakeSession())
    assert voice.config_path == config_path
    voice.close()
