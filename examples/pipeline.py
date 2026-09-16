"""Render multiple acoustic variants from one semantic UtterancePlan."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "One semantic plan can be rendered more than once."


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan.save(artefact_path("pipeline.utterplan.json"))

    normal = pipeline.render_plan(plan)
    fast = pipeline.render_plan(plan, length_scale=0.85)
    quiet = pipeline.render_plan(plan, volume=0.50)

    assert normal.plan_id == plan.plan_id
    assert fast.plan_id == plan.plan_id
    assert quiet.plan_id == plan.plan_id

    normal.save_wav(artefact_path("pipeline_normal.wav"))
    fast.save_wav(artefact_path("pipeline_fast.wav"))
    quiet.save_wav(artefact_path("pipeline_quiet.wav"))

    print(f"Voice: {VOICE}")
    print(f"Plan: {plan.plan_id}")
    print("Rendered normal, fast, and quiet acoustic variants from the same plan.")
