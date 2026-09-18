from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from audiocompose import Composer
from piperg2p import PhonemeSentence, PhonemizeResult, VoiceConfig
from utterplan import PauseConfig

from pipersynth.config import PipelineConfig
from pipersynth.pipeline import PiperPipeline


class FakeFrontend:
    def phonemize_prepared(self, text, *, annotations=None):
        sentence = PhonemeSentence(tuple(text), tuple(1 for _ in text))
        return PhonemizeResult(text, (sentence,))


class FakeVoice:
    def __init__(self):
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
        self.calls = []

    def resolve_speaker_id(self, value):
        return None

    def synthesize_ids(self, ids, config):
        self.calls.append(tuple(ids))
        return np.ones(len(ids), dtype=np.float32)

    @property
    def diagnostics(self):
        from pipersynth.diagnostics import RuntimeDiagnostics

        return RuntimeDiagnostics(sample_rate=10, num_symbols=4, num_speakers=1)

    def close(self):
        pass


def make_pipeline(**kwargs):
    voice = FakeVoice()
    pipeline = PiperPipeline(
        PipelineConfig("voice.onnx", language="en-us", **kwargs),
        voice_factory=lambda _config: voice,
    )
    return pipeline, voice


def test_to_audio_job_maps_segment_and_provenance():
    pipeline, voice = make_pipeline()
    plan = pipeline.plan("ab")

    job = pipeline.to_audio_job(plan)

    assert [item.id for item in job.items] == [plan.segments[0].id]
    assert job.items[0].metadata["pipersynth.segment_id"] == plan.segments[0].id
    assert job.output.sample_rate == 10
    assert job.output.loudness.target_lufs is None
    assert job.output.loudness.true_peak_ceiling_dbtp is None
    assert job.producer["name"] == "pipersynth"
    assert job.source["utterplan.plan_id"] == plan.plan_id
    assert voice.calls == [(1, 1)]
    pipeline.close()


def test_to_audio_job_maps_positive_pauses_once():
    pipeline, _voice = make_pipeline()
    plan = pipeline.plan(
        "Hello. World.",
        unit="sentence",
        pauses=PauseConfig(mode="manual", sentence=0.2),
    )

    job = pipeline.to_audio_job(plan)

    kinds = [(item.id, type(item).__name__) for item in job.items]
    assert sum(kind == "Silence" for _item, kind in kinds) == 1
    assert kinds[1][0].startswith("pause:")
    assert all(item.seconds > 0 for item in job.items if type(item).__name__ == "Silence")
    pipeline.close()

def test_render_plan_composes_once_and_uses_composition_result(monkeypatch):
    from audiocompose import Composer as RealComposer

    import pipersynth.pipeline as pipeline_module

    calls = []
    composed = []

    class SpyComposer:
        def compose(self, job):
            calls.append(job)
            result = RealComposer().compose(job)
            composed.append(result)
            return result

    monkeypatch.setattr(pipeline_module, "Composer", SpyComposer)
    pipeline, _voice = make_pipeline()
    plan = pipeline.plan("ab")
    result = pipeline.render_plan(plan)

    assert len(calls) == 1
    np.testing.assert_array_equal(result.audio, composed[0].audio)
    assert result.sample_rate == composed[0].sample_rate
    assert result.timing.composition_ms is not None
    pipeline.close()


def test_render_plan_rebuilds_retained_units_from_composition():
    pipeline, _voice = make_pipeline(retain_unit_audio=True)
    plan = pipeline.plan("ab")
    result = pipeline.render_plan(plan)

    assert len(result.chunks) == 1
    np.testing.assert_array_equal(result.chunks[0].audio, result.audio)
    pipeline.close()


def test_composition_keeps_unresolved_markers_explicit():
    pipeline, _voice = make_pipeline(document_format="ssmd")
    plan = pipeline.plan("Hello @inside world")
    result = pipeline.render_plan(plan)

    assert result.markers
    assert result.markers[0]["timing"] == "unresolved"
    assert result.markers[0]["sample_offset"] is None
    pipeline.close()




def test_audio_job_save_load_replays_fake_audio(tmp_path: Path):
    pipeline, _voice = make_pipeline()
    plan = pipeline.plan("ab")
    job = pipeline.to_audio_job(plan)

    manifest = Path(job.save(tmp_path / "speech.audiojob"))
    loaded = type(job).load(manifest)

    assert manifest.name == "audiojob.json"
    assert loaded.to_dict(base_dir=str(manifest.parent))["producer"] == dict(job.producer)
    first = Composer().compose(job)
    replay = Composer().compose(loaded)
    np.testing.assert_array_equal(first.audio, replay.audio)
    assert json.dumps(loaded.to_dict(base_dir=str(manifest.parent)), allow_nan=False)
    pipeline.close()
