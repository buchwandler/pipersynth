"""Optional German example using a voice matched to the plan language."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_GERMAN_VOICE", "de_DE-thorsten-medium")
TEXT = (
    "Hallo aus PiperSynth. Dieser Satz wird zuerst als UtterPlan geplant und danach "
    "mit einer deutschen Piper-Stimme gerendert."
)


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
    language_aliases={"de-de": "de"},
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan.save(artefact_path("german.utterplan.json"))
    pipeline.render_plan(plan).save_wav(artefact_path("german.wav"))
    print(f"Rendered German plan with {VOICE}: {plan.plan_id}")
