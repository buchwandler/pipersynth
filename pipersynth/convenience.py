from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from .audio import write_wav
from .config import GenerationConfig
from .pipeline import PiperPipeline


def _save_wav_atomically(result: Any, destination: Path) -> None:
    if destination.exists() and destination.is_dir():
        raise ValueError(f"output path is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_wav(temporary, result.audio, result.sample_rate)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def synthesize_to_wav(
    text: str,
    output: str | Path,
    *,
    voice: str,
    speaker: int | str | None = None,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
    sentence_silence: float = 0.0,
    volume: float = 1.0,
    normalize_audio: bool = True,
    providers: Sequence[Any] | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    text_preparation: Literal["identity", "spokenform"] = "identity",
    language: str | None = None,
    progress: Any | None = None,
) -> Path:
    """Download or reuse a catalog voice and atomically write a mono PCM WAV."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(voice, str) or not voice:
        raise ValueError("voice must be a non-empty string")
    destination = Path(output)
    if destination == Path(".") or not str(destination):
        raise ValueError("output must be a file path")
    generation = GenerationConfig(
        speaker=speaker,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=normalize_audio,
        volume=volume,
        sentence_silence=sentence_silence,
    )
    with PiperPipeline.from_pretrained(
        voice,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        generation=generation,
        providers=providers,
        text_preparation=text_preparation,
        language=language,
        progress=progress,
    ) as pipeline:
        result = pipeline.run(text)
        _save_wav_atomically(result, destination)
    return destination


def synthesize(
    text: str,
    *,
    voice: str,
    speaker: int | str | None = None,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
    sentence_silence: float = 0.0,
    volume: float = 1.0,
    normalize_audio: bool = True,
    providers: Sequence[Any] | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    text_preparation: Literal["identity", "spokenform"] = "identity",
    language: str | None = None,
    progress: Any | None = None,
) -> Any:
    """Synthesize an in-memory AudioResult with a managed catalog voice."""

    generation = GenerationConfig(
        speaker=speaker,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=normalize_audio,
        volume=volume,
        sentence_silence=sentence_silence,
    )
    with PiperPipeline.from_pretrained(
        voice,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        generation=generation,
        providers=providers,
        text_preparation=text_preparation,
        language=language,
        progress=progress,
    ) as pipeline:
        return pipeline.run(text)
