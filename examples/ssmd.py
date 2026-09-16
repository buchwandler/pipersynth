"""Use SSMD directives supported by the PiperSynth renderer."""

from __future__ import annotations

import os

from _output import artefact_path

from pipersynth import PiperPipeline

VOICE = os.environ.get("PIPERSYNTH_EXAMPLE_VOICE", "en_US-lessac-medium")
SOURCE = (
    'Welcome ...500ms [tomato]{ph="təˈmeɪtoʊ"}. '
    '@checkpoint [This part is slightly faster]{rate="1.15" volume="90%"}.'
)


with PiperPipeline.from_pretrained(
    VOICE,
    document_format="ssmd",
    text_preparation="spokenform",
    directive_policy="error",
    unit="sentence",
) as pipeline:
    plan = pipeline.plan(SOURCE, unit="sentence")
    plan.save(artefact_path("ssmd.utterplan.json"))

    print(f"Annotations: {plan.annotations}")
    print(f"Markers: {plan.markers}")
    for segment in plan.segments:
        print(f"Directives for {segment.id}: {segment.directives}")
        print(
            f"Resolved pauses: before={segment.pause_before.seconds:.3f}s "
            f"after={segment.pause_after.seconds:.3f}s"
        )

    # Pitch, emphasis, external audio, and model voice switching are excluded.
    # PiperSynth does not currently render those directives.
    pipeline.render_plan(plan).save_wav(artefact_path("ssmd.wav"))
