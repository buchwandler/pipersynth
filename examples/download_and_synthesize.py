"""Resolve a catalog voice automatically and render a persisted UtterPlan."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import ConsoleAssetProgress, PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "This example resolves a Piper voice automatically before rendering its plan."


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
    progress=ConsoleAssetProgress(),
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("download_and_synthesize.utterplan.json")
    plan.save(plan_path)

    result = pipeline.render_plan(plan)
    wav_path = artefact_path("download_and_synthesize.wav")
    result.save_wav(wav_path)

    print(f"Resolved voice: {VOICE}")
    print(f"Plan: {plan.plan_id}")
    print(f"Plan path: {plan_path}")
    print(f"WAV path: {wav_path}")
