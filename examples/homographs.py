"""Optional homograph stress test for the English Piper voice."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = (
    "Please record the new record. "
    "Do not desert us in the desert. "
    "Wind the rope while the wind is calm. "
    "I will read the report that I read yesterday."
)


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan.save(artefact_path("homographs.utterplan.json"))
    pipeline.render_plan(plan).save_wav(artefact_path("homographs.wav"))
    print(f"Rendered homographs plan: {plan.plan_id}")
