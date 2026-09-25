#!/usr/bin/env python3
"""Show PiperG2P rendering punctuation in already-prepared text."""

from __future__ import annotations

import os

from pipersynth import PiperVoice

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "A short sentence, a question? And an exclamation!"


with PiperVoice.from_pretrained(VOICE) as voice:
    result = voice.synthesize_text(TEXT, language="en-us")

wav_path = artefact_path("punctuation.wav")
result.save_wav(wav_path)
print(f"WAV path: {wav_path}")
