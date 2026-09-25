from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .asset_progress import AssetProgressEvent
from .errors import InvalidSynthesisConfigError
from .session import ProviderConfig, ProviderSpec
from .types import RenderedSegment, SynthesisConfig
from .voice import PiperVoice
from .voice_level import VoiceLevelConfig


def _synthesis_config(
    *,
    length_scale: float | None,
    noise_scale: float | None,
    noise_w_scale: float | None,
    normalize_audio: bool,
    output_gain: float,
    voice_level: VoiceLevelConfig | None,
) -> SynthesisConfig:
    return SynthesisConfig(
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=normalize_audio,
        output_gain=output_gain,
        voice_level=voice_level or VoiceLevelConfig(),
    )


def _render_prepared_text(
    prepared_text: str,
    *,
    voice: str,
    language: str,
    id: str | None,
    speaker: int | str | None,
    config: SynthesisConfig,
    providers: Sequence[ProviderSpec | ProviderConfig] | None,
    provider_options: dict[str, Any] | None,
    session_options: Any | None,
    frontend_options: dict[str, Any] | None,
    cache_dir: str | Path | None,
    offline: bool | None,
    refresh_catalog: bool,
    force_download: bool,
    progress: Callable[[AssetProgressEvent], None] | None,
) -> RenderedSegment:
    if not isinstance(prepared_text, str):
        raise TypeError("prepared_text must be a string")
    if not isinstance(voice, str) or not voice:
        raise ValueError("voice must be a non-empty string")
    if not isinstance(language, str) or not language:
        raise ValueError("language must be a non-empty string")
    with PiperVoice.from_pretrained(
        voice,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        frontend_options=frontend_options,
        progress=progress,
    ) as engine:
        return engine.synthesize_text(
            prepared_text,
            language=language,
            id=id,
            speaker=speaker,
            config=config,
        )


def synthesize(
    prepared_text: str,
    *,
    voice: str,
    language: str,
    id: str | None = None,
    speaker: int | str | None = None,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
    normalize_audio: bool = True,
    output_gain: float = 1.0,
    voice_level: VoiceLevelConfig | None = None,
    providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
    provider_options: dict[str, Any] | None = None,
    session_options: Any | None = None,
    frontend_options: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: Callable[[AssetProgressEvent], None] | None = None,
) -> RenderedSegment:
    """Synthesize prepared speakable text with one managed PiperVoice."""
    config = _synthesis_config(
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=normalize_audio,
        output_gain=output_gain,
        voice_level=voice_level,
    )
    return _render_prepared_text(
        prepared_text,
        voice=voice,
        language=language,
        id=id,
        speaker=speaker,
        config=config,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        frontend_options=frontend_options,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        progress=progress,
    )


def _save_wav_atomically(result: RenderedSegment, destination: Path) -> None:
    if destination.exists() and destination.is_dir():
        raise ValueError(f"output path is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        result.save_wav(temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def synthesize_to_wav(
    prepared_text: str,
    destination: str | Path,
    *,
    voice: str,
    language: str,
    id: str | None = None,
    speaker: int | str | None = None,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
    normalize_audio: bool = True,
    output_gain: float = 1.0,
    voice_level: VoiceLevelConfig | None = None,
    providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
    provider_options: dict[str, Any] | None = None,
    session_options: Any | None = None,
    frontend_options: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: Callable[[AssetProgressEvent], None] | None = None,
) -> Path:
    """Synthesize prepared speakable text and atomically save a mono PCM WAV."""
    path = Path(destination)
    if path == Path(".") or not str(path):
        raise InvalidSynthesisConfigError("destination must be a file path")
    result = synthesize(
        prepared_text,
        voice=voice,
        language=language,
        id=id,
        speaker=speaker,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=normalize_audio,
        output_gain=output_gain,
        voice_level=voice_level,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        frontend_options=frontend_options,
        cache_dir=cache_dir,
        offline=offline,
        refresh_catalog=refresh_catalog,
        force_download=force_download,
        progress=progress,
    )
    _save_wav_atomically(result, path)
    return path
