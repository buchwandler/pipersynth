#!/usr/bin/env python3
"""Synthesize one prepared text request with a catalog Piper voice."""

from __future__ import annotations

import os

from pipersynth import PiperVoice

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "Hello from PiperSynth. This is prepared, speakable text."


with PiperVoice.from_pretrained(VOICE) as voice:
    result = voice.synthesize_text(TEXT, language="en-us")

wav_path = artefact_path("basic.wav")
result.save_wav(wav_path)
print(f"Rendered {result.id} with {VOICE}: {wav_path}")
