#!/usr/bin/env python3
"""Synthesize prepared German text with a Piper German voice."""

from __future__ import annotations

import os

from pipersynth import PiperVoice

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_GERMAN_VOICE", "de_DE-thorsten-medium")
TEXT = "Guten Morgen. Dies ist vorbereiteter, sprechbarer Text."


with PiperVoice.from_pretrained(VOICE) as voice:
    result = voice.synthesize_text(TEXT, language="de")

wav_path = artefact_path("german.wav")
result.save_wav(wav_path)
print(f"WAV path: {wav_path}")
