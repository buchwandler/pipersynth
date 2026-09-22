import math

import numpy as np

from pipersynth import (
    LoudnessConfig,
    VoiceCalibrationKey,
    apply_voice_level_calibration,
    default_voice_calibration,
)

FAILED_KEYS = {
    "piper:ar_JO-kareem-low:low:speaker-0",
    "piper:ar_JO-kareem-medium:medium:speaker-0",
    "piper:he_IL-saspeech-medium:medium:speaker-0",
    "piper:ja_JP-hi_fi_captain-medium:medium:speaker-0",
    "piper:ja_JP-hi_fi_captain-medium:medium:speaker-1",
    "piper:lt_LT-reginute1-medium:medium:speaker-0",
    "piper:th_TH-tsync2-medium:medium:speaker-0",
    "piper:zh_CN-chaowen-medium:medium:speaker-0",
    "piper:zh_CN-xiao_ya-medium:medium:speaker-0",
}


def test_packaged_catalog_has_complete_measured_data():
    catalog = default_voice_calibration()

    assert catalog.schema == 1
    assert catalog.method == "bs1770"
    assert catalog.corpus == "pipersynth-count-1-to-10-v1"
    assert catalog.reference_lufs == -24.0
    assert len(catalog.voices) == 2704
    assert [str(key) for key in catalog.voices] == sorted(str(key) for key in catalog.voices)
    assert not FAILED_KEYS & {str(key) for key in catalog.voices}

    for record in catalog.voices.values():
        assert math.isfinite(record.gain_db)
        assert math.isfinite(record.measured_lufs)
        assert math.isfinite(record.reference_lufs)
        assert math.isfinite(record.mad_lu)
        assert record.samples == 3
        assert record.method == "bs1770"
        assert record.corpus_version == "pipersynth-count-1-to-10-v1"


def test_packaged_catalog_contains_representative_measured_identities():
    catalog = default_voice_calibration()
    keys = [
        "piper:en_US-lessac-medium:medium:speaker-0",
        "piper:en_US-arctic-medium:medium:speaker-1",
        "piper:vi_VN-25hours_single-low:low:speaker-0",
        "piper:de_DE-pavoque-low:low:speaker-0",
        "piper:en_US-libritts_r-medium:medium:speaker-761",
        "piper:bn_BD-google-medium:medium:speaker-2",
    ]

    assert all(VoiceCalibrationKey.parse(key) in catalog.voices for key in keys)
    assert catalog.voices[VoiceCalibrationKey.parse(keys[-1])].mad_lu > 0.75
    assert catalog.voices[VoiceCalibrationKey.parse(keys[4])].gain_db < -12.0


def test_runtime_lookup_uses_packaged_data_and_preserves_override_precedence():
    catalog = default_voice_calibration()
    key = VoiceCalibrationKey.parse("piper:en_US-lessac-medium:medium:speaker-0")
    record = catalog.voices[key]
    audio = np.array([0.25, -0.25], dtype=np.float32)

    leveled, application = apply_voice_level_calibration(
        audio, LoudnessConfig(voice_leveling="calibrated"), key
    )
    assert application.source == "catalog"
    assert application.key == key
    assert application.gain_db == record.gain_db
    np.testing.assert_allclose(leveled, audio * 10 ** (record.gain_db / 20))

    overridden, application = apply_voice_level_calibration(
        audio, LoudnessConfig(voice_leveling="calibrated", voice_gain_db=-1.0), key
    )
    assert application.source == "override"
    assert application.gain_db == -1.0
    np.testing.assert_allclose(overridden, audio * 10 ** (-1.0 / 20))

    unchanged, application = apply_voice_level_calibration(audio, LoudnessConfig(), key)
    assert application.source == "off"
    assert application.gain_db == 0.0
    np.testing.assert_array_equal(unchanged, audio)


def test_failed_identity_uses_missing_calibration_runtime_path():
    key = VoiceCalibrationKey.parse("piper:ar_JO-kareem-low:low:speaker-0")
    audio = np.array([0.25, -0.25], dtype=np.float32)

    result, application = apply_voice_level_calibration(
        audio, LoudnessConfig(voice_leveling="calibrated"), key
    )

    assert application.source == "missing_calibration"
    assert application.gain_db == 0.0
    assert application.key == key
    np.testing.assert_array_equal(result, audio)
