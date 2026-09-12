from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

AssetProgressPhase = Literal[
    "catalog-load",
    "catalog-refresh",
    "voice-resolve",
    "download-start",
    "download-complete",
    "cache-hit",
]


@dataclass(frozen=True, slots=True)
class AssetProgressEvent:
    phase: AssetProgressPhase
    voice_id: str | None = None
    message: str | None = None


AssetProgressCallback = Callable[[AssetProgressEvent], None]


class ConsoleAssetProgress:
    """Simple opt-in progress reporter for command-line use."""

    def __call__(self, event: AssetProgressEvent) -> None:
        detail = f": {event.message}" if event.message else ""
        voice = f" [{event.voice_id}]" if event.voice_id else ""
        print(f"{event.phase}{voice}{detail}")
