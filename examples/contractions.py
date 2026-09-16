"""Smoke-test contractions downstream of semantic planning and Piper G2P."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = (
    "I've learned a lot. You've taught me well. "
    "She'd believed we'd succeed, and they wouldn't give up."
)


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan.save(artefact_path("contractions.utterplan.json"))
    pipeline.render_plan(plan).save_wav(artefact_path("contractions.wav"))
    print(f"Rendered contractions plan: {plan.plan_id}")
