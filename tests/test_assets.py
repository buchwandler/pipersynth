from pathlib import Path

from pipersynth.assets import VoiceBundle


def test_voice_bundle_preserves_model_card(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    (tmp_path / "voice.onnx.json").write_text("{}")
    (tmp_path / "MODEL_CARD").write_text("license details")
    bundle = VoiceBundle.from_directory(tmp_path)
    assert bundle.model_path == model
    assert bundle.model_card_text == "license details"
