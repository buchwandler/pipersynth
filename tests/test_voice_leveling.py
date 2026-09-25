from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from piperg2p import VoiceConfig

from pipersynth import (
    PiperVoice,
    SynthesisConfig,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
    VoiceLevelConfig,
    apply_voice_level_calibration,
    load_voice_calibration,
)
from pipersynth.voice_level import CalibrationDataError


def voice_config(num_speakers: int = 1) -> VoiceConfig:
    return VoiceConfig.from_dict(
        {
            "num_symbols": 4,
            "num_speakers": num_speakers,
            "audio": {"sample_rate": 22050},
            "phoneme_type": "text",
            "phoneme_id_map": {"_": [0], "a": [1], "b": [2], " ": [3]},
            "speaker_id_map": {"alice": 1} if num_speakers > 1 else {},
            "default_speaker_id": 1 if num_speakers > 1 else 0,
        }
    )


def test_catalog_gain_and_explicit_override_precedence() -> None:
    key = VoiceCalibrationKey("piper", "voice", "medium", "speaker-0")
    catalog = VoiceCalibrationCatalog(
        1,
        "bs1770",
        "test",
        -24.0,
        {"test": "1"},
        {key: VoiceLevelCalibration(-6.0, samples=3)},
    )
    audio = np.array([1.0], dtype=np.float32)
    leveled, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), key, catalog=catalog
    )
    assert application.source == "catalog"
    np.testing.assert_allclose(leveled, [10 ** (-6 / 20)], rtol=1e-5)
    overridden, application = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated", gain_db=-12.0),
        key,
        catalog=catalog,
    )
    assert application.source == "override"
    np.testing.assert_allclose(overridden, [10 ** (-12 / 20)], rtol=1e-5)


def test_piper_voice_applies_explicit_engine_gain_and_output_gain() -> None:
    class Runtime:
        def infer(self, ids, **kwargs):
            return SimpleNamespace(
                audio=np.array([[[0.2, -0.4]]], dtype=np.float32), sample_rate=22050
            )

    config = voice_config()
    voice = PiperVoice(Runtime(), config, frontend=object())  # type: ignore[arg-type]
    audio = voice.synthesize_ids(
        [1],
        config=SynthesisConfig(
            normalize_audio=False,
            output_gain=0.5,
            voice_level=VoiceLevelConfig(mode="calibrated", gain_db=-6.0),
        ),
    )
    np.testing.assert_allclose(audio, np.array([0.2, -0.4]) * 0.5 * 10 ** (-6 / 20))
    assert voice.last_voice_level_application is not None
    assert voice.last_voice_level_application.source == "override"


def test_managed_calibration_key_uses_resolved_speaker_name() -> None:
    config = voice_config(2)
    installation = SimpleNamespace(id="voice", metadata={"quality": "medium"})
    voice = PiperVoice(
        SimpleNamespace(infer=lambda *args, **kwargs: None),
        config,
        frontend=object(),  # type: ignore[arg-type]
        installation=installation,
    )
    assert voice.calibration_key("alice") == VoiceCalibrationKey(
        "piper", "voice", "medium", "speaker-1"
    )


def test_loader_rejects_duplicate_keys_unknown_fields_and_nonfinite(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 1}')
    with pytest.raises(CalibrationDataError):
        load_voice_calibration(duplicate)

    base = {
        "schema": 1,
        "method": "bs1770",
        "corpus": "test",
        "reference_lufs": -24.0,
        "generated_with": {"test": "1"},
        "voices": {"piper:voice:medium:speaker-0": {"gain_db": 0.0}},
    }
    unknown = tmp_path / "unknown.json"
    payload = json.loads(json.dumps(base))
    payload["voices"]["piper:voice:medium:speaker-0"]["extra"] = 1
    unknown.write_text(json.dumps(payload))
    with pytest.raises(CalibrationDataError):
        load_voice_calibration(unknown)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(json.dumps({**base, "reference_lufs": float("nan")}))
    with pytest.raises(CalibrationDataError):
        load_voice_calibration(nonfinite)
