#!/usr/bin/env python3
"""Write one complete atomic Piper result to a mono WAV file."""

from __future__ import annotations

import os
import wave

from pipersynth import PiperVoice, SynthesisRequest
from pipersynth.audio import audio_to_int16_bytes

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
request = SynthesisRequest(
    id="atomic-request",
    text="The first sentence is ready. The second follows in the same request.",
    language="en-us",
)
wav_path = artefact_path("stream.wav")
with PiperVoice.from_pretrained(VOICE) as voice:
    result = voice.synthesize(request)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(result.sample_rate)
        wav.writeframes(audio_to_int16_bytes(result.audio))

print(f"Wrote atomic request {request.id} to {wav_path}")
