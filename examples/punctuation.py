"""Exercise punctuation handling through the semantic planner."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
TEXT = (
    '"Well," said the professor, "this is unusual!"\n\n'
    "The experiment, which took years, produced three results: "
    "95%, 87%, and 72%.\n\n"
    '"But wait..." she asked, "are you sure?"'
)


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="plain",
    text_preparation="spokenform",
) as pipeline:
    plan = pipeline.plan(TEXT, unit="sentence")
    plan.save(artefact_path("punctuation.utterplan.json"))

    for segment in plan.segments:
        print(
            f"segment={segment.id} text={segment.text!r} "
            f"before={segment.pause_before.seconds:.3f}s "
            f"after={segment.pause_after.seconds:.3f}s"
        )

    pipeline.render_plan(plan).save_wav(artefact_path("punctuation.wav"))
