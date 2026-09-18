from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._onnxvoice import adapt_progress, normalize_piper_ref
from .asset_progress import AssetProgressCallback
from .assets import VoiceBundle, VoiceMetadata
from .errors import (
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    CatalogUnavailableError,
    OfflineAssetError,
    OptionalDependencyError,
    VoiceNotFoundError,
)


@dataclass(frozen=True, slots=True)
class CacheInfo:
    directory: Path
    catalog_path: Path
    cached_voices: tuple[str, ...]
    catalog_cached: bool


def _offline_value(value: bool | None) -> bool:
    if value is not None:
        return value
    return os.environ.get("PIPERSYNTH_OFFLINE", "").casefold() in {"1", "true", "yes", "on"}

def _translate_asset_error(exc: Exception) -> AssetError:
    if isinstance(exc, AssetError):
        return exc
    name = type(exc).__name__
    if name == "OfflineError":
        return OfflineAssetError(str(exc))
    if name in {"IntegrityError", "ManifestError", "UnsafePathError", "LockError"}:
        return AssetCacheError(str(exc))
    if name in {"AssetNotFoundError", "NotInstalledError"}:
        return VoiceNotFoundError(str(exc))
    if name in {"CatalogError"}:
        return CatalogUnavailableError(str(exc))
    return AssetDownloadError(str(exc))


class VoiceAssetManager:
    """Compatibility facade delegating Piper assets to OnnxVoice."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        catalog_path: str | Path | None = None,
        offline: bool | None = None,
        progress: AssetProgressCallback | None = None,
    ) -> None:
        configured_cache = cache_dir or os.environ.get("PIPERSYNTH_CACHE_DIR")
        self.cache_dir = Path(configured_cache) if configured_cache else None
        self.catalog_path = (
            Path(catalog_path)
            if catalog_path is not None
            else (self.cache_dir / "catalogs" / "piper.json" if self.cache_dir else Path(""))
        )
        self._explicit_catalog = catalog_path is not None
        self.offline = _offline_value(offline)
        self.progress = progress

    def _manager(self, *, catalog: bool = False) -> Any:
        try:
            import onnxvoice
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "OnnxVoice is required for Piper asset management. Install pipersynth[catalog]."
            ) from exc
        sources = {"piper": str(self.catalog_path)} if catalog else None
        return onnxvoice.OnnxVoice(
            cache_dir=self.cache_dir,
            catalog_sources=sources,
            offline=self.offline,
        )

    def _metadata_from_item(self, item: Any) -> VoiceMetadata:
        raw = dict(getattr(item, "metadata", {}) or {})
        language = raw.get("language") or {}
        entry = {
            "id": getattr(item, "id", ""),
            "name": raw.get("name") or getattr(item, "id", ""),
            "language": language,
            "quality": raw.get("quality") or "",
            "num_speakers": raw.get("num_speakers") or 1,
            "speaker_id_map": raw.get("speaker_id_map") or {},
            "aliases": getattr(item, "aliases", ()),
        }
        return VoiceMetadata.from_catalog_entry(
            entry,
            source_revision=str(raw.get("source_revision") or "unknown"),
        )

    def list_voices(
        self,
        *,
        language: str | None = None,
        quality: str | None = None,
        refresh: bool = False,
    ) -> tuple[VoiceMetadata, ...]:
        try:
            items = self._manager(catalog=self._explicit_catalog).list(
                "piper",
                language=language,
                quality=quality,
                refresh=refresh,
                progress=adapt_progress(self.progress),
            )
        except Exception as exc:
            raise _translate_asset_error(exc) from exc
        return tuple(self._metadata_from_item(item) for item in items)

    def get_voice_metadata(self, voice: str, *, refresh: bool = False) -> VoiceMetadata:
        normalized = normalize_piper_ref(voice)
        item_id = normalized.split(":", 1)[1]
        for metadata in self.list_voices(refresh=refresh):
            if metadata.id == item_id or item_id in metadata.aliases:
                return metadata
        raise VoiceNotFoundError(f"Unknown Piper voice: {voice}")

    def resolve_voice(
        self,
        voice: str,
        *,
        download: bool = True,
        refresh_catalog: bool = False,
        force_download: bool = False,
    ) -> VoiceBundle:
        manager = self._manager(catalog=self._explicit_catalog)
        try:
            if not download:
                installation = manager.resolve(normalize_piper_ref(voice))
            else:
                installation = manager.install(
                    normalize_piper_ref(voice),
                    refresh=refresh_catalog,
                    force=force_download,
                    progress=adapt_progress(self.progress),
                )
        except Exception as exc:
            raise _translate_asset_error(exc) from exc
        return VoiceBundle.from_installation(installation)

    def is_voice_cached(self, voice: str) -> bool:
        try:
            self._manager().resolve(normalize_piper_ref(voice))
        except Exception:
            return False
        return True

    def voice_cache_path(self, voice: str) -> Path:
        return self._manager().where(normalize_piper_ref(voice))

    def cached_voices(self) -> tuple[VoiceBundle, ...]:
        return tuple(VoiceBundle.from_installation(item) for item in self._manager().installed("piper"))

    def remove_voice(self, voice: str) -> None:
        self._manager().remove(normalize_piper_ref(voice))

    def prune(self) -> tuple[Path, ...]:
        manager = self._manager()
        known = {item.id for item in manager.list("piper")}
        removed: list[Path] = []
        for installation in manager.installed("piper"):
            if installation.id not in known:
                removed.append(installation.path)
                manager.remove(installation.ref)
        return tuple(removed)

    def clear(self, *, voices: bool = False) -> None:
        if self.cache_dir is None:
            return
        root = self.cache_dir / "installs" / "piper" if voices else self.cache_dir
        shutil.rmtree(root, ignore_errors=True)

    def cache_info(self) -> CacheInfo:
        directory = self.cache_dir or Path.home() / ".cache" / "onnxvoice"
        catalog_path = self.catalog_path if str(self.catalog_path) else directory / "catalogs" / "piper.json"
        return CacheInfo(
            directory=directory,
            catalog_path=catalog_path,
            cached_voices=tuple(bundle.voice_id for bundle in self.cached_voices() if bundle.voice_id),
            catalog_cached=catalog_path.exists(),
        )


def list_voices(
    *,
    language: str | None = None,
    quality: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh: bool = False,
) -> tuple[VoiceMetadata, ...]:
    return VoiceAssetManager(cache_dir, offline=offline).list_voices(
        language=language, quality=quality, refresh=refresh
    )


def list_cached_voices(
    *,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
) -> tuple[VoiceBundle, ...]:
    return VoiceAssetManager(cache_dir, offline=offline).cached_voices()
