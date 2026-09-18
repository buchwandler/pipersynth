from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from piperg2p import PiperFrontend, VoiceConfig

import pipersynth._onnxvoice as boundary
from pipersynth import PiperVoice, SynthesisConfig
from pipersynth.errors import ModelInferenceError, VoiceNotFoundError


def voice_config() -> VoiceConfig:
    return VoiceConfig.from_dict(
        {
            "num_symbols": 8,
            "num_speakers": 1,
            "audio": {"sample_rate": 22050},
            "phoneme_type": "text",
            "espeak": {"voice": "en-us"},
            "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]},
        }
    )


class FakeRuntime:
    def __init__(self, *, sample_rate: int = 22050, failure: Exception | None = None):
        self.sample_rate = sample_rate
        self.failure = failure
        self.calls = []
        self.session = SimpleNamespace(
            provider_request=("CPUExecutionProvider",),
            resolved_providers=("CPUExecutionProvider",),
            input_names=("input", "input_lengths", "scales"),
            output_names=("audio",),
        )

    def infer(self, token_ids, **kwargs):
        self.calls.append((tuple(token_ids), kwargs))
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            audio=np.asarray([0.25, -0.5], dtype=np.float32),
            sample_rate=self.sample_rate,
            timings=np.zeros((1, 2), dtype=np.float32),
            outputs={"duration": np.zeros((1, 2), dtype=np.float32)},
        )

    def close(self):
        pass


def test_voice_passes_piper_policy_to_onnxvoice_and_summarizes_outputs():
    config = voice_config()
    runtime = FakeRuntime()
    voice = PiperVoice(runtime, config, PiperFrontend(config))

    inference = voice._infer_ids(
        [1, 3],
        SynthesisConfig(
            length_scale=0.8,
            noise_scale=0.4,
            noise_w_scale=0.2,
            normalize_audio=False,
        ),
    )

    assert runtime.calls == [
        (
            (1, 3),
            {"speaker_id": None, "noise_scale": 0.4, "length_scale": 0.8, "noise_w": 0.2},
        )
    ]
    assert inference.timing_summary == {"shape": [1, 2], "dtype": "float32"}
    assert inference.output_summary == {"duration": {"shape": [1, 2], "dtype": "float32"}}
    assert voice._infer_ids([]).audio.size == 0
    assert len(runtime.calls) == 1


def test_voice_wraps_inference_failure_and_preserves_cause():
    error = RuntimeError("runtime failure")
    voice = PiperVoice(FakeRuntime(failure=error), voice_config(), PiperFrontend(voice_config()))

    with pytest.raises(ModelInferenceError) as raised:
        voice.synthesize_ids([1, 3])

    assert raised.value.__cause__ is error


def test_voice_rejects_native_sample_rate_mismatch():
    voice = PiperVoice(
        FakeRuntime(sample_rate=16000), voice_config(), PiperFrontend(voice_config())
    )

    with pytest.raises(ModelInferenceError, match="sample rate"):
        voice.synthesize_ids([1, 3])


def test_local_boundary_passes_piper_system_and_cpu_default(monkeypatch, tmp_path: Path):
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    model.write_bytes(b"model")
    config.write_text("{}")
    runtime = object()
    calls = []

    fake_module = SimpleNamespace(
        open_local=lambda **kwargs: calls.append(kwargs) or runtime,
    )
    monkeypatch.setattr(boundary, "_onnxvoice", lambda: fake_module)

    assert boundary.open_local_voice(model, config) is runtime
    assert calls == [
        {
            "system": "piper",
            "model": model,
            "config": config,
            "providers": "CPUExecutionProvider",
            "provider_options": None,
            "session_options": None,
        }
    ]


def test_piper_ref_rejects_other_systems():
    with pytest.raises(VoiceNotFoundError):
        boundary.normalize_piper_ref("kokoro:model")


def test_piper_sources_do_not_own_onnxruntime_or_catalog_downloader():
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("pipersynth").glob("*.py"))
    assert "import onnxruntime" not in source
    assert "from onnxruntime" not in source
    assert "piper_voice_catalog" not in source


def test_lower_level_packages_do_not_depend_on_pipersynth_or_each_other():
    import audiocompose
    import onnxvoice

    for package in (onnxvoice, audiocompose):
        root = Path(package.__file__).parent
        source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
        assert "pipersynth" not in source
    compose_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path(audiocompose.__file__).parent).rglob("*.py")
    )
    assert "onnxvoice" not in compose_source
    assert "piperg2p" not in compose_source
    assert "utterplan" not in compose_source
