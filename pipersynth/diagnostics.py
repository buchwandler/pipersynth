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
    voice_id: str | None = None
    voice_source_revision: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic mapping."""
        return asdict(self)
