from __future__ import annotations

from pathlib import Path

from pipersynth.assets import VoiceBundle, VoiceMetadata
from pipersynth.pipeline import PiperPipeline


def test_from_pretrained_builds_config_and_infers_language(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    card = tmp_path / "MODEL_CARD"
    model.write_bytes(b"")
    config.write_text("{}")
    card.write_text("license")
    metadata = VoiceMetadata("voice", "Voice", "en_US", "en", "US", "medium", 1, {}, (), "a" * 40)
    bundle = VoiceBundle(model, config, card, metadata)

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve_voice(self, *args, **kwargs):
            return bundle

    import pipersynth.asset_manager as asset_manager

    monkeypatch.setattr(asset_manager, "VoiceAssetManager", Manager)
    pipeline = PiperPipeline.from_pretrained("voice", text_preparation="spokenform")
    assert pipeline.config.model_path == model
    assert pipeline.config.config_path == config
    assert pipeline.config.language == "en-us"
    assert pipeline.voice_bundle is bundle
    pipeline.close()


def test_explicit_language_wins(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    card = tmp_path / "MODEL_CARD"
    model.write_bytes(b"")
    config.write_text("{}")
    card.write_text("license")
    bundle = VoiceBundle(
        model,
        config,
        card,
        VoiceMetadata("voice", "Voice", "de_DE", "de", "DE", "medium", 1, {}, (), "a" * 40),
    )

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def resolve_voice(self, *args, **kwargs):
            return bundle

    import pipersynth.asset_manager as asset_manager

    monkeypatch.setattr(asset_manager, "VoiceAssetManager", Manager)
    pipeline = PiperPipeline.from_pretrained(
        "voice", text_preparation="spokenform", language="custom"
    )
    assert pipeline.config.language == "custom"
    pipeline.close()
