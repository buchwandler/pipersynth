import os
from pathlib import Path

import numpy as np
import pytest

from pipersynth import PiperVoice


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
