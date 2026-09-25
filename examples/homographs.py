#!/usr/bin/env python3
"""Supply a source-aligned pronunciation override for a prepared-text homograph."""

from __future__ import annotations

import os

from pipersynth import PiperVoice, PronunciationOverride, SynthesisSegment

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "I read the book yesterday."

segment = SynthesisSegment(
    id="read-past-tense",
    text=TEXT,
    language="en-us",
    pronunciation_overrides=(PronunciationOverride(2, 6, phonemes="ɹɛd"),),
)
with PiperVoice.from_pretrained(VOICE) as voice:
    result = voice.synthesize(segment)

wav_path = artefact_path("homographs.wav")
result.save_wav(wav_path)
print(f"WAV path: {wav_path}")
