from __future__ import annotations

from typing import Any

from audiocompose import CompositionResult
from utterplan import UtterancePlan

from .audio_job import PiperAudioJobContext
from .diagnostics import RuntimeDiagnostics, TimingDiagnostics
from .types import AudioChunk, AudioResult


def _marker_values(
    plan: UtterancePlan,
    composition: CompositionResult,
    context: PiperAudioJobContext,
) -> list[dict[str, Any]]:
    composed = {marker.id: marker for marker in composition.markers}
    unresolved = {marker["id"]: marker for marker in context.unresolved_markers}
    result: list[dict[str, Any]] = []
    for marker in plan.markers:
        value = composed.get(marker.id)
        if value is not None:
            result.append(
                {
                    "id": marker.id,
                    "name": marker.name,
                    "char_offset": marker.spoken_position,
                    "sample_offset": value.sample_offset,
                    "timing": "resolved",
                }
            )
        else:
            result.append(
                dict(
                    unresolved.get(
                        marker.id,
                        {
                            "id": marker.id,
                            "name": marker.name,
                            "char_offset": marker.spoken_position,
                            "sample_offset": None,
                            "timing": "unresolved",
                        },
                    )
                )
            )
    return result


def _chunks_from_composition(
    plan: UtterancePlan,
    composition: CompositionResult,
    context: PiperAudioJobContext,
) -> list[AudioChunk]:
    composed_items = {item.item_id: item for item in composition.items}
    chunks: list[AudioChunk] = []
    for record in context.units:
        spans = [
            composed_items[item_id]
            for item_id in record.item_ids
            if item_id in composed_items
        ]
        if spans:
            start = spans[0].start_sample
            end = spans[-1].end_sample
            audio = composition.audio[start:end]
        else:
            audio = composition.audio[:0]
        chunks.append(
            AudioChunk(
                sample_rate=composition.sample_rate,
                audio_float_array=audio,
                phonemes=record.phonemes,
                phoneme_ids=record.phoneme_ids,
                warnings=record.warnings,
                metadata={"item_ids": list(record.item_ids)},
            )
        )
    return chunks


def audio_result_from_composition(
    *,
    plan: UtterancePlan,
    context: PiperAudioJobContext,
    composition: CompositionResult,
    diagnostics: RuntimeDiagnostics | None,
    timing: TimingDiagnostics | None,
    retain_unit_audio: bool,
) -> AudioResult:
    metadata: dict[str, Any] = {
        "plan_id": plan.plan_id,
        "utterplan_producer": dict(plan.producer),
        "utterplan_schema_version": plan.schema_version,
        "voice_id": context.voice_id,
        "composition_provenance": dict(composition.provenance),
    }
    chunks = (
        _chunks_from_composition(plan, composition, context) if retain_unit_audio else []
    )
    return AudioResult(
        audio=composition.audio,
        sample_rate=composition.sample_rate,
        source_text=plan.source.text,
        prepared_text=plan.texts.spoken,
        plan=plan,
        plan_id=plan.plan_id,
        chunks=chunks,
        markers=_marker_values(plan, composition, context),
        warnings=tuple(plan.warnings) + context.warnings,
        diagnostics=diagnostics,
        timing=timing,
        metadata=metadata,
    )


__all__ = ["audio_result_from_composition"]
