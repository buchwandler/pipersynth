#!/usr/bin/env python3
"""Render one already-shaped long request without engine-side splitting."""

from __future__ import annotations

import os

from pipersynth import PiperVoice, SynthesisInputTooLongError, SynthesisRequest

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = (
    "The field team arrived after sunrise and inspected the instruments before "
    "beginning the survey. They recorded the temperature and compared each reading "
    "with the previous day's notes. At midday, the crew repeated the checks at the "
    "eastern station and reviewed the measurements before returning to camp."
)

request = SynthesisRequest(
    id="prepared-survey",
    text=TEXT,
    language="en-us",
)
with PiperVoice.from_pretrained(VOICE) as voice:
    try:
        result = voice.synthesize(request)
    except SynthesisInputTooLongError as error:
        print(
            "Atomic request exceeds known model capacity: "
            f"text={error.text_length}, phonemes={error.phoneme_count}, "
            f"maximum={error.max_phonemes}, model={error.model_id!r}. "
            "Choose a boundary in the caller before retrying."
        )
    else:
        wav_path = artefact_path("long_text.wav")
        result.save_wav(wav_path)
        print(f"Rendered one atomic request with {VOICE}: {wav_path}")
        print(f"Synthesis identity hash: {result.metadata['synthesis_hash']}")
