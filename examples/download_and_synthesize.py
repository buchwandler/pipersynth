#!/usr/bin/env python3
"""Install a catalog voice with progress and render a prepared sample."""

from __future__ import annotations

import os

from pipersynth import ConsoleAssetProgress, PiperVoice

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "PiperSynth uses OnnxVoice to install and run this Piper voice."


with PiperVoice.from_pretrained(VOICE, progress=ConsoleAssetProgress()) as voice:
    result = voice.synthesize_text(TEXT, language="en-us")

wav_path = artefact_path("download_and_synthesize.wav")
result.save_wav(wav_path)
print(f"WAV path: {wav_path}")
