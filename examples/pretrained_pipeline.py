"""Reuse one managed ONNX session for multiple explicitly planned utterances."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan_one = pipeline.plan("This is the first utterance.", unit="sentence")
    plan_one.save(artefact_path("pretrained_01.utterplan.json"))
    pipeline.render_plan(plan_one).save_wav(artefact_path("pretrained_01.wav"))

    plan_two = pipeline.plan(
        "The same ONNX session renders this second utterance.",
        unit="sentence",
    )
    plan_two.save(artefact_path("pretrained_02.utterplan.json"))
    pipeline.render_plan(plan_two).save_wav(artefact_path("pretrained_02.wav"))

    print(f"Voice: {VOICE}")
    print(f"Plans: {plan_one.plan_id}, {plan_two.plan_id}")
    print("Both plans were rendered by one managed pipeline session.")
