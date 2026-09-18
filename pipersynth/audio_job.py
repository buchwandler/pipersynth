from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    LoudnessPolicy,
    OutputPolicy,
    Silence,
)
from utterplan import UtterancePlan

from ._version import __version__
from .audio import postprocess_audio
from .plan_adapter import PreparedPiperSegment, PreparedPiperUnit


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class RenderedPiperSegment:
    segment_id: str
    unit_id: str
    spoken_start: int
    spoken_end: int
    audio: np.ndarray
    sample_rate: int
    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    pause_before_seconds: float
    pause_after_seconds: float
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PiperUnitAudioRecord:
    unit_id: str
    item_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    marker_ids: tuple[str, ...]
    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PiperAudioJobContext:
    job: AudioJob
    segments: tuple[RenderedPiperSegment, ...]
    units: tuple[PiperUnitAudioRecord, ...]
    unresolved_markers: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    voice_id: str | None


def render_segment(
    prepared: PreparedPiperSegment,
    *,
    unit_id: str,
    spoken_start: int,
    spoken_end: int,
    voice: Any,
) -> RenderedPiperSegment:
    inference_metadata: dict[str, Any] = {}
    if prepared.phoneme_ids:
        if hasattr(voice, "_infer_ids"):
            inference = voice._infer_ids(prepared.phoneme_ids, prepared.synthesis)
            audio = postprocess_audio(
                inference.audio,
                normalize=prepared.synthesis.normalize_audio,
                volume=prepared.synthesis.volume,
            )
            if inference.timing_summary is not None:
                inference_metadata["pipersynth.onnx_timings"] = _json_safe(
                    inference.timing_summary
                )
            if inference.output_summary:
                inference_metadata["pipersynth.onnx_outputs"] = _json_safe(
                    inference.output_summary
                )
        else:
            audio = voice.synthesize_ids(prepared.phoneme_ids, prepared.synthesis)
    else:
        audio = np.zeros(0, dtype=np.float32)
    metadata = {
        "pipersynth.frontend": _json_safe(prepared.metadata),
        **inference_metadata,
    }
    return RenderedPiperSegment(
        segment_id=prepared.plan_segment_id,
        unit_id=unit_id,
        spoken_start=spoken_start,
        spoken_end=spoken_end,
        audio=np.asarray(audio, dtype=np.float32),
        sample_rate=voice.config.sample_rate,
        phonemes=prepared.phonemes,
        phoneme_ids=prepared.phoneme_ids,
        pause_before_seconds=prepared.pause_before_seconds,
        pause_after_seconds=prepared.pause_after_seconds,
        warnings=prepared.warnings,
        metadata=metadata,
    )


def _anchors_for_segment(
    segment: RenderedPiperSegment,
    markers: Sequence[Any],
) -> tuple[tuple[AudioAnchor, ...], set[str]]:
    anchors: list[AudioAnchor] = []
    resolved: set[str] = set()
    for marker in markers:
        if marker.spoken_position == segment.spoken_start:
            offset = 0
        elif marker.spoken_position == segment.spoken_end:
            offset = int(segment.audio.size)
        else:
            continue
        anchors.append(AudioAnchor(marker.id, offset, marker.name))
        resolved.add(marker.id)
    return tuple(anchors), resolved


def build_audio_job_context(
    *,
    plan: UtterancePlan,
    prepared_units: Sequence[PreparedPiperUnit],
    voice: Any,
    voice_id: str | None,
    producer_version: str = __version__,
) -> PiperAudioJobContext:
    segments_by_id = {segment.id: segment for segment in plan.segments}
    markers_by_id = {marker.id: marker for marker in plan.markers}
    items: list[AudioClip | Silence] = []
    rendered_segments: list[RenderedPiperSegment] = []
    unit_records: list[PiperUnitAudioRecord] = []
    unresolved: list[dict[str, Any]] = []
    warnings: list[str] = []

    for unit in prepared_units:
        item_ids: list[str] = []
        unit_phonemes: list[str] = []
        unit_ids: list[int] = []
        unit_warnings: list[str] = []
        for prepared in unit.segments:
            plan_segment = segments_by_id[prepared.plan_segment_id]
            rendered = render_segment(
                prepared,
                unit_id=unit.plan_unit_id,
                spoken_start=plan_segment.spoken_start,
                spoken_end=plan_segment.spoken_end,
                voice=voice,
            )
            rendered_segments.append(rendered)
            unit_phonemes.extend(rendered.phonemes)
            unit_ids.extend(rendered.phoneme_ids)
            unit_warnings.extend(rendered.warnings)
            warnings.extend(rendered.warnings)

            if rendered.pause_before_seconds > 0:
                pause_id = f"pause:{rendered.segment_id}:before"
                items.append(
                    Silence(
                        pause_id,
                        rendered.pause_before_seconds,
                        {"pipersynth.segment_id": rendered.segment_id, "pipersynth.position": "before"},
                    )
                )
                item_ids.append(pause_id)

            if rendered.audio.size:
                anchors, resolved = _anchors_for_segment(
                    rendered,
                    [markers_by_id[marker_id] for marker_id in unit.marker_ids],
                )
                clip_metadata = {
                    "pipersynth.segment_id": rendered.segment_id,
                    "pipersynth.unit_id": unit.plan_unit_id,
                    "pipersynth.voice": voice_id,
                    "pipersynth.language": prepared.language,
                    "pipersynth.phonemes": list(rendered.phonemes),
                    "pipersynth.phoneme_ids": list(rendered.phoneme_ids),
                    "pipersynth.warnings": list(rendered.warnings),
                    **_json_safe(rendered.metadata),
                }
                items.append(
                    AudioClip(
                        rendered.segment_id,
                        AudioBufferSource(rendered.audio, rendered.sample_rate),
                        anchors=anchors,
                        metadata=clip_metadata,
                    )
                )
                item_ids.append(rendered.segment_id)
                for marker_id in resolved:
                    markers_by_id.pop(marker_id, None)

            if rendered.pause_after_seconds > 0:
                pause_id = f"pause:{rendered.segment_id}:after"
                items.append(
                    Silence(
                        pause_id,
                        rendered.pause_after_seconds,
                        {"pipersynth.segment_id": rendered.segment_id, "pipersynth.position": "after"},
                    )
                )
                item_ids.append(pause_id)

        for marker_id in unit.marker_ids:
            marker = markers_by_id.get(marker_id)
            if marker is not None:
                unresolved.append(
                    {
                        "id": marker.id,
                        "name": marker.name,
                        "char_offset": marker.spoken_position,
                        "sample_offset": None,
                        "timing": "unresolved",
                    }
                )
        unit_records.append(
            PiperUnitAudioRecord(
                unit.plan_unit_id,
                tuple(item_ids),
                unit.segment_ids,
                unit.marker_ids,
                tuple(unit_phonemes),
                tuple(unit_ids),
                tuple(unit_warnings),
            )
        )

    output = OutputPolicy(
        sample_rate=voice.config.sample_rate,
        channels=1,
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        clip_policy="clamp",
    )
    producer = {"name": "pipersynth", "version": producer_version}
    source = {
        "utterplan.plan_id": plan.plan_id,
        "utterplan.schema_version": plan.schema_version,
        "utterplan.producer": _json_safe(dict(plan.producer)),
    }
    job = AudioJob(tuple(items), output=output, producer=producer, source=source)
    return PiperAudioJobContext(
        job=job,
        segments=tuple(rendered_segments),
        units=tuple(unit_records),
        unresolved_markers=tuple(unresolved),
        warnings=tuple(warnings),
        voice_id=voice_id,
    )


__all__ = [
    "PiperAudioJobContext",
    "PiperUnitAudioRecord",
    "RenderedPiperSegment",
    "build_audio_job_context",
    "render_segment",
]
