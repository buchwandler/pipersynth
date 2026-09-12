import os
import wave
from pathlib import Path

import numpy as np
import pytest

from pipersynth import PiperVoice, synthesize_to_wav


@pytest.mark.integration
def test_real_voice_smoke_when_asset_is_configured() -> None:
    model = os.environ.get("PIPERSYNTH_VOICE_MODEL")
    if not model:
        pytest.skip("set PIPERSYNTH_VOICE_MODEL to run a real voice smoke test")
    runtime = pytest.importorskip("onnxruntime")
    del runtime
    with PiperVoice.load(Path(model)) as voice:
        chunks = list(voice.synthesize("Hello world."))
    assert chunks
    assert chunks[0].audio.dtype == np.float32
    assert chunks[0].audio.ndim == 1
    assert np.all(np.isfinite(chunks[0].audio))


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.resource_heavy
def test_catalog_voice_download_and_offline_reuse(tmp_path: Path, monkeypatch) -> None:
    if not os.environ.get("PIPERSYNTH_RUN_CATALOG_INTEGRATION"):
        pytest.skip("set PIPERSYNTH_RUN_CATALOG_INTEGRATION=1 to run the catalog test")
    pytest.importorskip("onnxruntime")
    cache_dir = tmp_path / "cache"
    output = synthesize_to_wav(
        "Hello from PiperSynth.",
        tmp_path / "hello.wav",
        voice="en_US-lessac-medium",
        cache_dir=cache_dir,
    )
    assert output.exists() and output.stat().st_size > 44
    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() > 0
        assert wav.getnframes() > 0
    offline_output = synthesize_to_wav(
        "Hello from PiperSynth.",
        tmp_path / "offline.wav",
        voice="en_US-lessac-medium",
        cache_dir=cache_dir,
        offline=True,
    )
    assert offline_output.exists()
