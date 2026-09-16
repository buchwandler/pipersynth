from __future__ import annotations

import numpy as np
import pytest
from piperg2p import PhonemeSentence, PhonemizeResult, VoiceConfig
from ttsplan import PauseConfig, PlanValidationError

from pipersynth import PiperPipeline
from pipersynth.config import GenerationConfig, PipelineConfig
from pipersynth.planning import planner_config_from_pipersynth


def test_planner_config_mapping_and_legacy_pause() -> None:
    config = PipelineConfig(
        "voice.onnx",
        language="EN_US",
        generation=GenerationConfig(sentence_silence=0.2),
        pauses=PauseConfig(sentence=0.4),
        document_format="ssmd",
        text_preparation="spokenform",
        unit="sentence",
    )
    planner = planner_config_from_pipersynth(config)
    assert planner.language == "en-us"
    assert planner.document_format == "ssmd"
    assert planner.text_preparation == "spokenform"
    assert planner.unit == "sentence"
    assert planner.pauses.sentence == 0.2



class FakeFrontend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def phonemize_prepared(self, text: str, *, annotations=None) -> PhonemizeResult:
        self.calls.append((text, annotations))
        sentence = PhonemeSentence(tuple(text), tuple(1 for _ in text))
        return PhonemizeResult(clean_text=text, sentences=(sentence,))


class FakeVoice:
    def __init__(self) -> None:
        self.config = VoiceConfig.from_dict(
            {
                "num_symbols": 4,
                "num_speakers": 1,
                "audio": {"sample_rate": 10},
                "phoneme_type": "text",
                "phoneme_id_map": {"a": [0], "b": [1], "_": [2], " ": [3]},
                "espeak_voice": "en-us",
            }
        )
        self.frontend = FakeFrontend()
        self.calls: list[tuple[int, ...]] = []
        self.closed = False

    def resolve_speaker_id(self, value):
        assert value is None or value == 0
        return None

    def synthesize_ids(self, ids, config):
        self.calls.append(tuple(ids))
        return np.ones(len(ids), dtype=np.float32)

    @property
    def diagnostics(self):
        from pipersynth.diagnostics import RuntimeDiagnostics

        return RuntimeDiagnostics(sample_rate=10, num_symbols=4, num_speakers=1)

    def close(self) -> None:
        self.closed = True

    def warmup(self) -> None:
        pass


def make_pipeline(voice: FakeVoice | None = None, **config_kwargs: object) -> tuple[PiperPipeline, FakeVoice]:
    active = voice or FakeVoice()
    pipeline = PiperPipeline(
        PipelineConfig("voice.onnx", language="en-us", **config_kwargs),
        voice_factory=lambda _config: active
    )
    return pipeline, active


def test_plan_is_lazy_and_render_plan_does_not_replan() -> None:
    pipeline, voice = make_pipeline()
    plan = pipeline.plan("ab")
    assert not voice.frontend.calls
    pipeline._planner.plan = lambda *args, **kwargs: pytest.fail("render_plan replanned")
    result = pipeline.render_plan(plan)
    assert result.plan is plan
    assert result.plan_id == plan.plan_id
    assert result.diagnostics is not None
    assert result.diagnostics.plan_id == plan.plan_id
    assert result.metadata["ttsplan_schema_version"] == plan.schema_version
    assert voice.frontend.calls[0][0] == plan.segments[0].text
    pipeline.close()
def test_pronunciation_directive_uses_raw_piper_path() -> None:
    pipeline, voice = make_pipeline(document_format="ssmd", text_preparation="spokenform")
    plan = pipeline.plan('[tomato]{ph="təˈmeɪtoʊ"}')
    pipeline.render_plan(plan)
    assert voice.frontend.calls[0][0] == "[[təˈmeɪtoʊ]]"
    pipeline.close()


def test_unsupported_directive_is_explicit() -> None:
    pipeline, _voice = make_pipeline(document_format="ssmd", text_preparation="spokenform")
    plan = pipeline.plan('[x]{pitch="+2st"}')
    with pytest.raises(Exception, match="pitch"):
        pipeline.render_plan(plan)
    pipeline.close()


def test_resolved_sentence_pause_is_rendered_once() -> None:
    pipeline, _voice = make_pipeline()
    plan = pipeline.plan(
        "One. Two.",
        unit="sentence",
        pauses=PauseConfig(mode="manual", sentence=0.2),
    )
    result = pipeline.render_plan(plan)
    assert result.audio.size == 10
    pipeline.close()




def test_plan_units_define_streaming_groups() -> None:
    pipeline, _voice = make_pipeline()
    prepared = pipeline.prepare_units("One. Two.", unit="sentence")
    assert [unit.unit_kind for unit in prepared.units] == ["sentence", "sentence"]
    assert all(unit.plan_unit_id for unit in prepared.units)
    prepared.close()
    pipeline.close()


def test_direct_phoneme_mode_bypasses_planner() -> None:
    pipeline, voice = make_pipeline()
    pipeline._planner.plan = lambda *args, **kwargs: pytest.fail("direct mode planned")
    result = pipeline.run("ab", is_phonemes=True)
    assert result.plan is None
    assert result.plan_id is None
    assert voice.calls
    pipeline.close()


def test_render_plan_rejects_planning_overrides() -> None:
    pipeline, _voice = make_pipeline()
    plan = pipeline.plan("ab")
    with pytest.raises(TypeError, match="planning override"):
        pipeline.render_plan(plan, document_format="ssmd")
    pipeline.close()


def test_raw_blocks_remain_explicit_with_spokenform() -> None:
    voice = FakeVoice()
    pipeline = PiperPipeline(
        PipelineConfig(
            "voice.onnx", language="en-us", text_preparation="spokenform"
        ),
        voice_factory=lambda _config: voice,
    )
    plan = pipeline.plan("Use 2 kg [[ tɛst ]] and 3 kg.")
    assert "[[ tɛst ]]" in plan.texts.spoken
    pipeline.render_plan(plan)
    assert any("[[" in text for text, _annotations in voice.frontend.calls)
    pipeline.close()


def test_render_plan_rejects_malformed_plan() -> None:
    pipeline, _voice = make_pipeline()
    plan = pipeline.plan("ab")
    from dataclasses import replace

    malformed = replace(plan, plan_id="invalid")
    with pytest.raises(PlanValidationError):
        pipeline.render_plan(malformed)
    pipeline.close()
