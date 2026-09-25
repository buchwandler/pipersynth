#!/usr/bin/env python3
"""Write request-local Piper sentence-group chunks incrementally to a WAV."""

from __future__ import annotations

import os
import wave

from pipersynth import PiperVoice, SynthesisSegment
from pipersynth.audio import audio_to_int16_bytes

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
segment = SynthesisSegment(
    id="streamed-request",
    text="The first sentence is ready. The second follows immediately.",
    language="en-us",
)

wav_path = artefact_path("stream.wav")
with PiperVoice.from_pretrained(VOICE) as voice:
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(voice.config.sample_rate)
        for chunk in voice.iter_chunks(segment):
            wav.writeframes(audio_to_int16_bytes(chunk.audio))

print(f"Streamed request {segment.id} to {wav_path}")
