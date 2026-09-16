"""Persist an UtterancePlan, load it again, and render the loaded plan."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline, UtterancePlan

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "This plan crosses a persistence boundary before it is rendered."


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    original = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("plan_roundtrip.utterplan.json")
    original.save(plan_path)

    loaded = UtterancePlan.load(plan_path)
    assert loaded == original
    assert loaded.plan_id == original.plan_id

    pipeline.render_plan(loaded).save_wav(artefact_path("plan_roundtrip.wav"))
    print(f"Round-tripped plan: {loaded.plan_id}")
