"""Build and replay a generic AudioJob from a PiperSynth plan."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory

from _output import artefact_path
from audiocompose import AudioJob, Composer, write_wav

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = "This AudioJob was produced by PiperSynth and replayed by AudioCompose."


with PiperPipeline.from_pretrained(VOICE) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan_path = artefact_path("audio_job.utterplan.json")
    plan.save(plan_path)

    with TemporaryDirectory() as job_directory:
        manifest = pipeline.to_audio_job(plan).save(job_directory)
        job = AudioJob.load(manifest)
        composition = Composer().compose(job)
        wav_path = artefact_path("audio_job.wav")
        write_wav(wav_path, composition.audio, composition.sample_rate)

    print(f"Voice: {VOICE}")
    print(f"Plan: {plan_path}")
    print("AudioJob: replayed from a temporary manifest")
    print(f"WAV path: {wav_path}")
