#!/usr/bin/env python3
"""Synthesize prepared long text and report request-local source ranges."""

from __future__ import annotations

import os

from pipersynth import PiperVoice, TextChunkingConfig

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = (
    "The field team arrived shortly after sunrise and checked the instruments "
    "before beginning the survey. They recorded the air temperature, inspected "
    "the sample containers, and compared each reading with the previous day's "
    "notes. A light wind moved across the ridge, but the equipment remained "
    "stable throughout the first round of measurements.\n\n"
    "At midday, the team repeated the checks at the eastern station. The new "
    "readings were close to the morning values, so the crew continued toward "
    "the river crossing. They described the route, the condition of the trail, "
    "and the locations where the signal briefly weakened. Every observation "
    "was entered as prepared speakable text before it was sent to PiperSynth.\n\n"
    "In the afternoon, the crew returned to the starting point and reviewed "
    "the final set of measurements. The equipment was packed away, and the "
    "team confirmed that all samples had been labeled correctly. This example "
    "uses request-local sentence and character-limit chunking only. It does "
    "not parse a document, plan semantic pauses, or create a timeline."
)

with PiperVoice.from_pretrained(VOICE) as voice:
    result = voice.synthesize_text(
        TEXT,
        language="en-us",
        chunking=TextChunkingConfig(max_chars=240),
    )

wav_path = artefact_path("long_text.wav")
result.save_wav(wav_path)
print(f"Rendered prepared long text with {VOICE}: {wav_path}")
for chunk in result.chunks:
    source = chunk.metadata["text_chunk"]
    print(
        f"chunk {chunk.index}: [{source['char_start']}:{source['char_end']}] "
        f"{TEXT[source['char_start'] : source['char_end']]!r}"
    )
print(f"Chunking summary: {result.metadata['text_chunking']}")
