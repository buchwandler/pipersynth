"""Render sentence units lazily from one persisted UtterancePlan."""

from __future__ import annotations

import os
import wave
from pathlib import Path

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "First sentence. Second sentence. Third sentence."


def save_unit_wav(path: Path, unit: object) -> None:
    """Write one AudioUnitResult as mono 16-bit PCM using the standard library."""
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(unit.sample_rate)  # type: ignore[attr-defined]
        stream.writeframes(unit.audio_int16_bytes)  # type: ignore[attr-defined]


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan.save(artefact_path("stream.utterplan.json"))

    with pipeline.prepare_plan(plan) as prepared:
        for descriptor in prepared.units:
            print(descriptor.index, descriptor.plan_unit_id, descriptor.content_hash)

        for unit in prepared.render():
            path = artefact_path(f"stream_sentence_{unit.descriptor.index + 1:03d}.wav")
            save_unit_wav(path, unit)
            print(
                f"unit={unit.descriptor.index} text={unit.descriptor.text!r} "
                f"phonemes={len(unit.phonemes)} duration={unit.duration_seconds:.3f}s "
                f"path={path}"
            )
