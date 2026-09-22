from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from piperg2p import PhonemeSentence, PhonemizeResult, VoiceConfig
from utterplan import UtterancePlan
from utterplan.units import make_units

from pipersynth.config import GenerationConfig, PipelineConfig
from pipersynth.errors import PlanRenderingError
from pipersynth.pipeline import PiperPipeline
from pipersynth.plan_adapter import prepare_plan


class AnnotationFrontend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def phonemize_prepared(self, text: str, *, annotations=None) -> PhonemizeResult:
        self.calls.append((text, annotations))
        values = tuple(annotations or ())
        tag = values[0].get("tag") if values else None
        phoneme = "V" if tag == "VBP" else "A" if tag == "JJ" else "x"
        identifier = 1 if phoneme == "V" else 2 if phoneme == "A" else 3
        sentence = PhonemeSentence((phoneme,), (identifier,))
        return PhonemizeResult(clean_text=text, sentences=(sentence,))


class NoAnnotationFrontend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def phonemize_prepared(self, text: str) -> PhonemizeResult:
        self.calls.append(text)
        sentence = PhonemeSentence(("x",), (1,))
        return PhonemizeResult(clean_text=text, sentences=(sentence,))


class FakeVoice:
    def __init__(self, frontend: Any) -> None:
        self.config = VoiceConfig.from_dict(
            {
                "num_symbols": 4,
                "num_speakers": 1,
                "audio": {"sample_rate": 10},
                "phoneme_type": "text",
                "phoneme_id_map": {"x": [0], "V": [1], "A": [2], "_": [3]},
                "espeak_voice": "en-us",
            }
        )
        self.frontend = frontend

    def resolve_speaker_id(self, value):
        assert value is None or value == 0
        return None


def make_plan(
    text: str,
    *,
    document_format: str = "plain",
    text_preparation: str = "identity",
) -> UtterancePlan:
    pipeline = PiperPipeline(
        PipelineConfig(
            "voice.onnx",
            language="en-us",
            document_format=document_format,
            text_preparation=text_preparation,
        )
    )
    try:
        return pipeline.plan(text)
    finally:
        pipeline.close()


def replace_tokens(plan: UtterancePlan, **changes: object) -> UtterancePlan:
    tokens = tuple(replace(token, **changes) for token in plan.tokens)
    units = make_units(plan.segments, plan.markers, tokens, plan.units[0].kind)
    return replace(plan, tokens=tokens, units=units).with_identity()


def replace_plan_layout(plan: UtterancePlan, *, segments=None, tokens=None) -> UtterancePlan:
    resolved_segments = tuple(segments or plan.segments)
    resolved_tokens = tuple(tokens or plan.tokens)
    units = make_units(resolved_segments, plan.markers, resolved_tokens, plan.units[0].kind)
    return replace(
        plan, segments=resolved_segments, tokens=resolved_tokens, units=units
    ).with_identity()


def render(plan: UtterancePlan, frontend: Any):
    return prepare_plan(plan, FakeVoice(frontend), GenerationConfig())


def test_v2_annotations_forward_all_fields_and_provenance() -> None:
    plan = replace_tokens(
        make_plan("live"),
        pos="VERB",
        tag="VBP",
        lemma="live",
        language="en-us",
        morph="Tense=Pres|VerbForm=Fin",
    )
    frontend = AnnotationFrontend()
    original_tokens = plan.tokens
    original_runs = plan.linguistic_runs
    original_membership = tuple(segment.token_indices for segment in plan.segments)

    prepared = render(plan, frontend)

    assert frontend.calls == [
        (
            "live",
            (
                {
                    "start": 0,
                    "end": 4,
                    "text": "live",
                    "pos": "VERB",
                    "tag": "VBP",
                    "lemma": "live",
                    "language": "en-us",
                    "morph": "Tense=Pres|VerbForm=Fin",
                },
            ),
        )
    ]
    assert prepared[0].segments[0].metadata["utterplan_linguistics"] == (
        {
            "language_run_id": "lang-0",
            "provider": "fallback",
            "model": None,
            "provider_version": None,
            "model_version": None,
            "token_range": (0, 1),
        },
    )
    assert plan.tokens == original_tokens
    assert plan.linguistic_runs == original_runs
    assert tuple(segment.token_indices for segment in plan.segments) == original_membership


def test_grammatical_annotations_select_distinct_prepared_phonemes() -> None:
    verb_plan = replace_tokens(
        make_plan("I live here."),
        pos="VERB",
        tag="VBP",
        morph="Tense=Pres|VerbForm=Fin",
    )
    adjective_plan = replace_tokens(
        make_plan("a live show"),
        pos="ADJ",
        tag="JJ",
        morph="Degree=Pos",
    )
    verb_frontend = AnnotationFrontend()
    adjective_frontend = AnnotationFrontend()

    verb_prepared = render(verb_plan, verb_frontend)
    adjective_prepared = render(adjective_plan, adjective_frontend)

    assert verb_prepared[0].segments[0].phonemes == ("V",)
    assert adjective_prepared[0].segments[0].phonemes == ("A",)
    assert verb_prepared[0].segments[0].phoneme_ids != adjective_prepared[0].segments[0].phoneme_ids


def test_incompatible_frontend_cannot_drop_non_empty_annotations() -> None:
    plan = replace_tokens(make_plan("live"), tag="VBP")

    with pytest.raises(PlanRenderingError, match="cannot accept Utterplan token annotations"):
        render(plan, NoAnnotationFrontend())


def test_annotation_free_segment_uses_annotation_free_frontend_path() -> None:
    plan = make_plan("live")
    segment = replace(plan.segments[0], token_indices=())
    plan = replace_plan_layout(plan, segments=(segment,))
    frontend = NoAnnotationFrontend()

    render(plan, frontend)

    assert frontend.calls == ["live"]


def test_cross_boundary_token_coordinates_are_clipped() -> None:
    plan = make_plan("abcd")
    segment = replace(plan.segments[0], text="bc", spoken_start=1, spoken_end=3)
    plan = replace_plan_layout(plan, segments=(segment,))
    frontend = AnnotationFrontend()

    render(plan, frontend)

    annotation = frontend.calls[0][1][0]
    assert annotation["start"] == 0
    assert annotation["end"] == 2


def test_explicit_pronunciation_remains_authoritative() -> None:
    plan = replace_tokens(
        make_plan(
            '[tomato]{ph="təˈmeɪtoʊ"}', document_format="ssmd", text_preparation="spokenform"
        ),
        pos="NOUN",
        tag="NN",
        morph="Number=Sing",
    )
    frontend = AnnotationFrontend()

    render(plan, frontend)

    assert frontend.calls == [("[[təˈmeɪtoʊ]]", None)]
