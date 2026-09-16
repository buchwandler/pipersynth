"""Render explicit and automatic pauses resolved by UtterPlan."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PauseConfig, PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
SOURCE_EXPLICIT = "Start ...500ms continue ...1s finish."
SOURCE_AUTO = (
    "The backup battery (still warm from the morning test) "
    "sat beside the console. Then the team continued."
)


def print_pause_details(plan: object) -> None:
    """Print boundaries and resolved segment pauses from an UtterPlan."""
    for boundary in plan.boundaries:  # type: ignore[attr-defined]
        print(f"boundary={boundary.id} kind={boundary.kind} seconds={boundary.seconds}")
    for segment in plan.segments:  # type: ignore[attr-defined]
        print(
            f"segment={segment.id} before={segment.pause_before.seconds:.3f}s "
            f"after={segment.pause_after.seconds:.3f}s"
        )


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="ssmd",
    text_preparation="spokenform",
) as pipeline:
    explicit_plan = pipeline.plan(
        SOURCE_EXPLICIT,
        unit="sentence",
        pauses=PauseConfig(mode="manual"),
    )
    explicit_plan.save(artefact_path("pauses_explicit.utterplan.json"))
    print("Explicit SSMD pauses")
    print_pause_details(explicit_plan)
    pipeline.render_plan(explicit_plan).save_wav(artefact_path("pauses_explicit.wav"))

    auto_plan = pipeline.plan(
        SOURCE_AUTO,
        unit="sentence",
        document_format="plain",
        pauses=PauseConfig(mode="auto", parenthetical=0.20, sentence=0.50),
    )
    auto_plan.save(artefact_path("pauses_auto.utterplan.json"))
    print("Automatic semantic pauses")
    print_pause_details(auto_plan)
    pipeline.render_plan(auto_plan).save_wav(artefact_path("pauses_auto.wav"))
