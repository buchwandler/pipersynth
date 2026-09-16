"""Smallest managed-voice, end-to-end UtterPlan example."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "Hello from PiperSynth. This audio was rendered from a saved UtterancePlan."


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("basic.utterplan.json")
    plan.save(plan_path)

    result = pipeline.render_plan(plan)
    wav_path = artefact_path("basic.wav")
    result.save_wav(wav_path)

    print(f"Voice: {VOICE}")
    print(f"Plan: {plan.plan_id}")
    print(f"Spoken text: {plan.texts.spoken}")
    print(f"Plan path: {plan_path}")
    print(f"WAV path: {wav_path}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Duration: {result.duration_seconds:.3f}s")
