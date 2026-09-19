import json
from types import SimpleNamespace

import numpy as np
import pytest
from piperg2p import PiperFrontend, VoiceConfig

from pipersynth import (
    LoudnessConfig,
    PiperVoice,
    SynthesisConfig,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
    apply_voice_level_calibration,
    load_voice_calibration,
)
from pipersynth.voice_level import CalibrationDataError


def _config(num_speakers=1):
    return VoiceConfig.from_dict(
        {
            "num_symbols": 4,
            "num_speakers": num_speakers,
            "audio": {"sample_rate": 22050},
            "phoneme_type": "text",
            "phoneme_id_map": {"_": [0], "a": [1], "b": [2], " ": [3]},
            "default_speaker_id": 1 if num_speakers > 1 else 0,
        }
    )


def test_exact_catalog_gain_and_override_precedence():
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
        audio, LoudnessConfig(voice_leveling="calibrated"), key, catalog=catalog
    )
    assert application.source == "catalog"
    assert np.isclose(leveled[0], 10 ** (-6 / 20), rtol=1e-5)
    overridden, application = apply_voice_level_calibration(
        audio,
        LoudnessConfig(voice_leveling="calibrated", voice_gain_db=-12.0),
        key,
        catalog=catalog,
    )
    assert application.source == "override"
    assert np.isclose(overridden[0], 10 ** (-12 / 20), rtol=1e-5)


def test_local_voice_has_no_catalog_identity_and_explicit_gain_still_works():
    runtime = SimpleNamespace(infer=lambda *args, **kwargs: None)
    voice = PiperVoice(runtime, _config(), PiperFrontend(_config()))
    assert voice.calibration_key(SynthesisConfig()) is None
    audio, application = apply_voice_level_calibration(
        np.array([1.0], dtype=np.float32),
        LoudnessConfig(voice_leveling="calibrated"),
        None,
    )
    assert application.source == "missing_identity"
    assert np.array_equal(audio, [1.0])


def test_managed_key_uses_numeric_speaker_and_quality():
    config = _config(3)
    installation = SimpleNamespace(id="voice", metadata={"quality": "medium"})
    runtime = SimpleNamespace(infer=lambda *args, **kwargs: None)
    voice = PiperVoice(runtime, config, PiperFrontend(config), installation=installation)
    assert voice.calibration_key(SynthesisConfig(speaker_id=1)) == VoiceCalibrationKey(
        "piper", "voice", "medium", "speaker-1"
    )
    assert voice.calibration_key(SynthesisConfig()) == VoiceCalibrationKey(
        "piper", "voice", "medium", "speaker-1"
    )


def test_loader_rejects_duplicate_keys_unknown_fields_and_nonfinite(tmp_path):
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


def test_processing_order_normalize_then_gain_then_volume():
    class Runtime:
        def infer(self, ids, **kwargs):
            return SimpleNamespace(
                audio=np.array([[[2.0, -1.0]]], dtype=np.float32), sample_rate=22050
            )

        def close(self):
            pass

    config = _config()
    voice = PiperVoice(Runtime(), config, PiperFrontend(config))
    audio = voice.synthesize_ids(
        [1],
        SynthesisConfig(
            volume=0.5,
            loudness=LoudnessConfig(voice_gain_db=-6.0),
        ),
    )
    assert np.allclose(audio, [10 ** (-6 / 20) * 0.5, -0.5 * 10 ** (-6 / 20) * 0.5])


def test_packaged_catalog_data_is_present():
    from importlib.resources import files

    assert files("pipersynth").joinpath("data", "voice_level_calibration.json").is_file()
