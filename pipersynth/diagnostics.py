from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    """Serializable metadata describing a PiperSynth runtime."""

    model_path: str | None = None
    config_path: str | None = None
    sample_rate: int | None = None
    num_symbols: int | None = None
    num_speakers: int | None = None
    speaker_names: tuple[str, ...] = ()
    phoneme_type: str | None = None
    espeak_voice: str | None = None
    providers_requested: tuple[str, ...] = ()
    providers_active: tuple[str, ...] = ()
    model_inputs: tuple[str, ...] = ()
    model_outputs: tuple[str, ...] = ()
    frontend: str | None = None
    plan_id: str | None = None
    utterplan_producer: dict[str, Any] | None = None
    utterplan_schema_version: int | None = None
    voice_id: str | None = None
    voice_source_revision: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic mapping."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingDiagnostics:
    """Optional stable timing measurements in milliseconds."""

    planning_ms: float | None = None
    g2p_ms: float | None = None
    inference_ms: float | None = None
    prepare_text_ms: float | None = None
    phonemize_ms: float | None = None
    postprocess_ms: float | None = None
    total_ms: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)
