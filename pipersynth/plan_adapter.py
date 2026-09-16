from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ttsplan import PlanSegment, TTSPlan, normalize_language

from .config import GenerationConfig
from .errors import (
    UnsupportedPlanDirectiveError,
    UnsupportedPlanLanguageError,
    VoiceBindingError,
)
from .types import SynthesisConfig


@dataclass(frozen=True, slots=True)
class PreparedPiperSegment:
    plan_segment_id: str
    text: str
    language: str
    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    pause_before_seconds: float
    pause_after_seconds: float
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any]
    synthesis: SynthesisConfig


@dataclass(frozen=True, slots=True)
class PreparedPiperUnit:
    plan_unit_id: str
    index: int
    kind: Literal["paragraph", "sentence"]
    spoken_start: int
    spoken_end: int
    segment_ids: tuple[str, ...]
    marker_ids: tuple[str, ...]
    content_hash: str
    segments: tuple[PreparedPiperSegment, ...]


def _policy_issue(policy: str, message: str) -> str | None:
    if policy == "error":
        raise UnsupportedPlanDirectiveError(message)
    if policy == "warn":
        return message
    return None


def _number(value: str, *, name: str) -> float:
    raw = value.strip()
    if raw.endswith("%"):
        return float(raw[:-1]) / 100.0
    return float(raw)


def _active_voice_id(voice: Any) -> str:
    bundle = getattr(voice, "voice_bundle", None)
    bundle_id = getattr(bundle, "voice_id", None)
    if bundle_id:
        return str(bundle_id)
    model_path = getattr(voice, "model_path", None)
    if model_path is not None:
        return str(model_path)
    return "active-piper-voice"


def _resolve_voice_speaker(
    plan: TTSPlan,
    segment: PlanSegment,
    voice: Any,
    generation: GenerationConfig,
) -> int | None:
    requested = segment.directives.voice
    if requested is None:
        return voice.resolve_speaker_id(generation.speaker)
    bindings = plan.document_metadata.get("voice_bindings", {})
    target = bindings.get(requested.reference, requested.reference) if isinstance(bindings, Mapping) else requested.reference
    active_id = _active_voice_id(voice)
    speaker = generation.speaker
    if target == active_id or target in {"active", "current"}:
        return voice.resolve_speaker_id(speaker)
    if isinstance(target, str) and target in getattr(voice.config, "speaker_id_map", {}):
        return voice.resolve_speaker_id(target)
    raise VoiceBindingError(
        f"voice binding {requested.reference!r} resolves to {target!r}, "
        f"but active Piper voice is {active_id!r}"
    )


def _check_language(plan: TTSPlan, segment: PlanSegment, voice: Any, aliases: Mapping[str, str]) -> str:
    language = normalize_language(segment.language, dict(aliases))
    voice_language = normalize_language(str(getattr(voice.config, "espeak_voice", "")), dict(aliases))
    if language != voice_language:
        raise UnsupportedPlanLanguageError(
            f"plan segment {segment.id} uses language {language!r}, but voice "
            f"{_active_voice_id(voice)!r} uses {voice_language!r}"
        )
    return language


def _segment_synthesis(
    plan: TTSPlan,
    segment: PlanSegment,
    voice: Any,
    generation: GenerationConfig,
    directive_policy: str,
) -> tuple[SynthesisConfig, list[str]]:
    warnings: list[str] = []
    speaker_id = _resolve_voice_speaker(plan, segment, voice, generation)
    length_scale = generation.length_scale
    volume = generation.volume
    prosody = segment.directives.prosody
    if prosody is not None:
        if prosody.rate is not None:
            try:
                rate = _number(prosody.rate, name="prosody.rate")
                if rate <= 0:
                    raise ValueError
                base = length_scale if length_scale is not None else voice.config.length_scale
                length_scale = base / rate
            except (TypeError, ValueError):
                warning = _policy_issue(directive_policy, f"unsupported prosody rate {prosody.rate!r}")
                if warning:
                    warnings.append(warning)
        if prosody.volume is not None:
            try:
                segment_volume = _number(prosody.volume, name="prosody.volume")
                if segment_volume < 0:
                    raise ValueError
                volume *= segment_volume
            except (TypeError, ValueError):
                warning = _policy_issue(directive_policy, f"unsupported prosody volume {prosody.volume!r}")
                if warning:
                    warnings.append(warning)
        if prosody.pitch is not None:
            warning = _policy_issue(directive_policy, "PiperSynth does not support pitch directives")
            if warning:
                warnings.append(warning)
    if segment.directives.emphasis is not None:
        warning = _policy_issue(directive_policy, "PiperSynth does not support emphasis directives")
        if warning:
            warnings.append(warning)
    if segment.directives.audio is not None:
        warning = _policy_issue(directive_policy, "PiperSynth does not support external audio directives")
        if warning:
            warnings.append(warning)
    return (
        SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=generation.noise_scale,
            noise_w_scale=generation.noise_w_scale,
            normalize_audio=generation.normalize_audio,
            volume=volume,
        ),
        warnings,
    )


def _token_annotations(plan: TTSPlan, segment: PlanSegment) -> tuple[dict[str, Any], ...]:
    result = []
    for index in segment.token_indices:
        token = plan.tokens[index]
        if token.spoken_start < segment.spoken_start or token.spoken_end > segment.spoken_end:
            continue
        result.append(
            {
                "start": token.spoken_start - segment.spoken_start,
                "end": token.spoken_end - segment.spoken_start,
                "text": token.text,
                "pos": token.pos,
                "tag": token.tag,
                "lemma": token.lemma,
                "language": token.language,
            }
        )
    return tuple(result)
def _sentences(result: Any) -> tuple[Any, ...]:
    sentences = tuple(getattr(result, "sentences", ()))
    if not sentences:
        legacy = tuple(getattr(result, "tokens", ()))
        if legacy and all(hasattr(item, "ids") for item in legacy):
            return legacy
    return sentences



def _phonemize(frontend: Any, text: str, annotations: tuple[dict[str, Any], ...]) -> Any:
    parameters = inspect.signature(frontend.phonemize_prepared).parameters
    if "annotations" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return frontend.phonemize_prepared(text, annotations=annotations)
    return frontend.phonemize_prepared(text)



def prepare_plan(
    plan: TTSPlan,
    voice: Any,
    generation: GenerationConfig,
    *,
    directive_policy: Literal["error", "warn", "ignore"] = "error",
    language_policy: Literal["strict", "allow"] = "strict",
    language_aliases: Mapping[str, str] | None = None
) -> tuple[PreparedPiperUnit, ...]:
    """Adapt a validated semantic plan into renderer-local Piper records."""

    plan.validate()
    aliases = dict(language_aliases or {})
    segments = {segment.id: segment for segment in plan.segments}
    prepared_units: list[PreparedPiperUnit] = []
    for unit in plan.units:
        prepared_segments: list[PreparedPiperSegment] = []
        for segment_id in unit.segment_ids:
            segment = segments[segment_id]
            language = (
                _check_language(plan, segment, voice, aliases)
                if language_policy == "strict"
                else normalize_language(segment.language, aliases)
            )
            synthesis, directive_warnings = _segment_synthesis(
                plan, segment, voice, generation, directive_policy
            )
            pronunciation = segment.directives.pronunciation
            if pronunciation is not None and pronunciation.alphabet not in {"ipa", "espeak-ipa3"}:
                warning = _policy_issue(
                    directive_policy,
                    f"unsupported pronunciation alphabet {pronunciation.alphabet!r}",
                )
                if warning:
                    directive_warnings.append(warning)
                pronunciation = None
            if pronunciation is not None:
                result = voice.frontend.phonemize_prepared(f"[[{pronunciation.phonemes}]]")
            else:
                result = _phonemize(
                    voice.frontend, segment.text, _token_annotations(plan, segment)
                )
            sentences = _sentences(result)
            phonemes = tuple(phone for sentence in sentences for phone in sentence.phonemes)
            phoneme_ids = tuple(identifier for sentence in sentences for identifier in sentence.ids)
            warnings = tuple(directive_warnings) + tuple(
                warning for sentence in sentences for warning in sentence.warnings
            ) + tuple(getattr(result, "warnings", ()))
            prepared_segments.append(
                PreparedPiperSegment(
                    plan_segment_id=segment.id,
                    text=segment.text,
                    language=language,
                    phonemes=phonemes,
                    phoneme_ids=phoneme_ids,
                    pause_before_seconds=segment.pause_before.seconds,
                    pause_after_seconds=segment.pause_after.seconds,
                    warnings=warnings,
                    metadata={"frontend_diagnostics": getattr(result, "diagnostics", None)},
                    synthesis=synthesis,
                )
            )
        prepared_units.append(
            PreparedPiperUnit(
                plan_unit_id=unit.id,
                index=unit.index,
                kind=unit.kind if unit.kind in {"sentence", "paragraph"} else "sentence",
                spoken_start=unit.spoken_start,
                spoken_end=unit.spoken_end,
                segment_ids=unit.segment_ids,
                marker_ids=unit.marker_ids,
                content_hash=unit.content_hash,
                segments=tuple(prepared_segments),
            )
        )
    return tuple(prepared_units)


__all__ = ["PreparedPiperSegment", "PreparedPiperUnit", "prepare_plan"]
