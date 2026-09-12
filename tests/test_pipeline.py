import numpy as np
from piperg2p import PhonemeSentence, PhonemizeResult, VoiceConfig

from pipersynth.config import GenerationConfig, PipelineConfig
from pipersynth.diagnostics import RuntimeDiagnostics
from pipersynth.pipeline import PiperPipeline


class FakeFrontend:
    def phonemize_prepared(self, text):
        sentences = tuple(
            PhonemeSentence(tuple(part), (1, 2)) for part in text.split("|") if part
        )
        return PhonemizeResult(text, sentences)


class FakeVoice:
    def __init__(self):
        self.config = VoiceConfig.from_dict(
            {
                "num_symbols": 4,
                "num_speakers": 1,
                "audio": {"sample_rate": 10},
                "phoneme_type": "text",
                "phoneme_id_map": {"a": [0], "b": [1], "_": [2], " ": [3]},
            }
        )
        self.frontend = FakeFrontend()
        self.calls = []
        self.warmups = 0
        self.closed = False

    def resolve_speaker_id(self, value):
        assert value is None or value == 0
        return None

    def synthesize_ids(self, ids, config):
        self.calls.append((tuple(ids), config))
        return np.asarray(ids, dtype=np.float32) / 2

    @property
    def diagnostics(self):
        return RuntimeDiagnostics(sample_rate=10, num_symbols=4, num_speakers=1)

    def warmup(self):
        self.warmups += 1

    def close(self):
        self.closed = True


def test_pipeline_is_lazy_reusable_and_supports_run_and_call():
    voices = []

    def factory(config):
        voice = FakeVoice()
        voices.append(voice)
        return voice

    pipeline = PiperPipeline(PipelineConfig("voice.onnx"), voice_factory=factory)
    assert not voices
    first = pipeline.run("ab|ba", volume=0.5)
    second = pipeline("ab")
    assert len(voices) == 1
    assert first.source_text == "ab|ba"
    assert first.prepared_text == "ab|ba"
    assert first.audio.shape == (4,)
    assert len(voices[0].calls) == 3
    assert pipeline.config.generation.volume == 1.0
    assert second.audio.shape == (2,)


def test_pipeline_iter_units_has_stable_indices_and_skip_support():
    voice = FakeVoice()
    pipeline = PiperPipeline(PipelineConfig("voice.onnx"), voice_factory=lambda config: voice)
    with pipeline.prepare_units("ab|ba|ab") as prepared:
        assert [unit.index for unit in prepared.units] == [0, 1, 2]
        rendered = list(prepared.render(skip_indices=(1,)))
    assert [unit.descriptor.index for unit in rendered] == [0, 2]
    streamed = list(pipeline.iter_units("ab|ba"))
    assert [unit.descriptor.index for unit in streamed] == [0, 1]


def test_pipeline_warmup_and_close_are_idempotent():
    voice = FakeVoice()
    pipeline = PiperPipeline(PipelineConfig("voice.onnx"), voice_factory=lambda config: voice)
    pipeline.warmup()
    pipeline.close()
    pipeline.close()
    assert voice.warmups == 1
    assert voice.closed


def test_generation_config_is_immutable_and_validates_silence():
    config = GenerationConfig(sentence_silence=0.1)
    assert config.sentence_silence == 0.1
    try:
        config.sentence_silence = 0.2
    except Exception:
        pass
    else:
        raise AssertionError("GenerationConfig must be immutable")
