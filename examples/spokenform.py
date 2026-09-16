"""Inspect written-to-spoken normalization captured in an UtterancePlan."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "Dr. Smith bought 5 kg of apples for $12.50. The meeting starts at 14:30 on 12/31/2026."


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    print(f"Source: {plan.source.text}")
    print(f"Structural: {plan.texts.structural}")
    print(f"Spoken: {plan.texts.spoken}")
    print(f"Replacements: {plan.preparation.replacements}")

    plan.save(artefact_path("spokenform.utterplan.json"))
    pipeline.render_plan(plan).save_wav(artefact_path("spokenform.wav"))
