from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .asset_progress import AssetProgressCallback, AssetProgressEvent
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


@contextmanager
def asset_lock(path: Path, *, timeout: float = 180.0) -> Iterator[None]:
    """Acquire a small cross-process lock using an atomically created file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, f"pid={os.getpid()}\ntime={time.time()}\n".encode())
            finally:
                os.close(descriptor)
            break
        except FileExistsError as exc:
            if _lock_is_stale(path):
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"Timed out waiting for asset lock: {path}") from exc
            time.sleep(0.05)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _lock_is_stale(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        pid = int(fields.get("pid", "0"))
        timestamp = float(fields.get("time", "0"))
    except (OSError, ValueError):
        return True
    if time.time() - timestamp > 180.0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class VoiceAssetManager:
    """Resolve Piper catalog metadata and downloaded voice artifacts."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        catalog_path: str | Path | None = None,
        offline: bool | None = None,
        progress: AssetProgressCallback | None = None,
    ) -> None:
        configured_cache = cache_dir or os.environ.get("PIPERSYNTH_CACHE_DIR")
        self.cache_dir = (
            Path(configured_cache) if configured_cache else user_cache_path("pipersynth")
        )
        self.catalog_path = (
            Path(catalog_path)
            if catalog_path is not None
            else self.cache_dir / "catalog" / "voices.json"
        )
        self._explicit_catalog = catalog_path is not None
        if offline is None:
            offline = os.environ.get("PIPERSYNTH_OFFLINE", "").casefold() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.offline = offline
        self.progress = progress

    @property
    def voices_dir(self) -> Path:
        return self.cache_dir / "voices"

    def _emit(self, phase: str, *, voice_id: str | None = None, message: str | None = None) -> None:
        if self.progress is not None:
            self.progress(AssetProgressEvent(phase, voice_id, message))  # type: ignore[arg-type]

    def _catalog_dependency(self) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        try:
            from piper_voice_catalog import (
                CatalogError,
                DownloadError,
                download_voice,
                fetch_and_build_catalog,
                get_voice,
                list_voices,
                load_catalog,
            )
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "catalog integration requires piper-onnx-voices. Install pipersynth[catalog]."
            ) from exc
        return (
            CatalogError,
            DownloadError,
            download_voice,
            fetch_and_build_catalog,
            get_voice,
            list_voices,
            load_catalog,
        )

    def _load_catalog(self, *, refresh: bool = False, emit: bool = True) -> dict[str, Any]:
        if refresh:
            if self.offline:
                raise OfflineAssetError("Cannot refresh the Piper voice catalog in offline mode")
            if self._explicit_catalog:
                raise CatalogUnavailableError("An explicit catalog path cannot be refreshed")
            self._refresh_catalog()
        if self.catalog_path.exists():
            try:
                _, _, _, _, _, _, load_catalog = self._catalog_dependency()
                catalog = load_catalog(self.catalog_path)
            except OptionalDependencyError:
                raise
            except Exception as exc:
                raise AssetCacheError(
                    f"Cached Piper voice catalog is invalid: {self.catalog_path}"
                ) from exc
            if emit:
                self._emit("catalog-load", message="Loaded cached Piper voice catalog")
            return catalog
        if self._explicit_catalog:
            if self.offline:
                raise OfflineAssetError(f"Piper voice catalog is not cached: {self.catalog_path}")
            raise CatalogUnavailableError(
                f"Piper voice catalog does not exist: {self.catalog_path}"
            )
        if self.offline:
            raise OfflineAssetError("Piper voice catalog is not cached and offline mode is enabled")
        self._refresh_catalog()
        return self._load_catalog()

    def _refresh_catalog(self) -> None:
        self._emit("catalog-refresh", message="Refreshing Piper voice catalog")
        try:
            _, _, _, fetch_and_build_catalog, _, _, _ = self._catalog_dependency()
            catalog = fetch_and_build_catalog()
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise CatalogUnavailableError("Unable to refresh the Piper voice catalog") from exc
        target = self.catalog_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with asset_lock(self.cache_dir / "locks" / "catalog.lock"):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="voices.", suffix=".tmp", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                temporary.write_text(
                    json.dumps(catalog, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
                )
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    def _metadata(self, entry: Mapping[str, Any], catalog: Mapping[str, Any]) -> VoiceMetadata:
        source = catalog.get("source")
        revision = source.get("revision", "unknown") if isinstance(source, Mapping) else "unknown"
        return VoiceMetadata.from_catalog_entry(entry, source_revision=str(revision))

    def list_voices(
        self,
        *,
        language: str | None = None,
        quality: str | None = None,
        refresh: bool = False,
    ) -> tuple[VoiceMetadata, ...]:
        catalog = self._load_catalog(refresh=refresh)
        try:
            _, _, _, _, _, list_voices, _ = self._catalog_dependency()
            entries = list_voices(catalog, language=language, quality=quality)
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise CatalogUnavailableError("Unable to list Piper voices") from exc
        return tuple(self._metadata(entry, catalog) for entry in entries)

    def get_voice_metadata(self, voice: str, *, refresh: bool = False) -> VoiceMetadata:
        catalog = self._load_catalog(refresh=refresh)
        try:
            _, _, _, _, get_voice, _, _ = self._catalog_dependency()
            entry = get_voice(catalog, voice)
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise VoiceNotFoundError(f"Unknown Piper voice: {voice}") from exc
        return self._metadata(entry, catalog)

    def is_voice_cached(self, voice: str) -> bool:
        try:
            metadata = self.get_voice_metadata(voice)
        except (AssetCacheError, CatalogUnavailableError, OfflineAssetError, VoiceNotFoundError):
            return False
        try:
            self._validated_bundle(metadata)
        except AssetCacheError:
            return False
        return True

    def voice_cache_path(self, voice: str) -> Path:
        return self.voices_dir / self.get_voice_metadata(voice).id

    def _validated_bundle(self, metadata: VoiceMetadata) -> VoiceBundle:
        target = self._safe_voice_path(metadata.id)
        try:
            bundle = VoiceBundle.from_directory(target, metadata=metadata)
        except Exception as exc:
            raise AssetCacheError(f"Cached Piper voice is invalid: {target}") from exc
        expected = {
            entry["filename"]: entry
            for entry in self._voice_entry(metadata.id)["artifacts"].values()
        }
        for path in (bundle.model_path, bundle.config_path, bundle.model_card):
            if path.is_symlink() or path.name not in expected:
                raise AssetCacheError(f"Cached Piper voice contains an unsafe artifact: {path}")
            artifact = expected[path.name]
            digest = md5(path.read_bytes(), usedforsecurity=False).hexdigest()
            if path.stat().st_size != artifact["size"] or digest != artifact["md5"]:
                raise AssetCacheError(f"Cached Piper voice artifact does not match catalog: {path}")
        return bundle

    def _voice_entry(self, voice_id: str) -> Mapping[str, Any]:
        catalog = self._load_catalog(emit=False)
        try:
            _, _, _, _, get_voice, _, _ = self._catalog_dependency()
            return get_voice(catalog, voice_id)
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise VoiceNotFoundError(f"Unknown Piper voice: {voice_id}") from exc

    def _safe_voice_path(self, voice_id: str) -> Path:
        root = self.voices_dir.resolve()
        target = (self.voices_dir / voice_id).resolve()
        if target.parent != root:
            raise AssetCacheError(f"Unsafe Piper voice cache path: {voice_id!r}")
        return target

    def resolve_voice(
        self,
        voice: str,
        *,
        download: bool = True,
        refresh_catalog: bool = False,
        force_download: bool = False,
    ) -> VoiceBundle:
        metadata = self.get_voice_metadata(voice, refresh=refresh_catalog)
        self._emit("voice-resolve", voice_id=metadata.id, message="Resolved Piper voice")
        target = self._safe_voice_path(metadata.id)
        if not force_download:
            try:
                bundle = self._validated_bundle(metadata)
            except AssetCacheError:
                if target.exists():
                    raise
            else:
                self._emit("cache-hit", voice_id=metadata.id, message="Using cached Piper voice")
                return bundle
        if not download or self.offline:
            raise OfflineAssetError(
                f"Piper voice is not cached and offline mode is enabled: {metadata.id}"
            )
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        with asset_lock(self.cache_dir / "locks" / f"voice-{metadata.id}.lock"):
            if not force_download:
                try:
                    return self._validated_bundle(metadata)
                except AssetCacheError:
                    pass
            self._emit("download-start", voice_id=metadata.id, message="Downloading Piper voice")
            try:
                _, download_error, download_voice, _, _, _, _ = self._catalog_dependency()
                download_voice(
                    dict(self._voice_entry(metadata.id)), target, overwrite=force_download
                )
            except OptionalDependencyError:
                raise
            except Exception as exc:
                if download_error is not None and isinstance(exc, download_error):
                    raise AssetDownloadError(
                        f"Unable to download Piper voice: {metadata.id}"
                    ) from exc
                raise AssetDownloadError(f"Unable to download Piper voice: {metadata.id}") from exc
            bundle = self._validated_bundle(metadata)
            self._emit("download-complete", voice_id=metadata.id, message="Downloaded Piper voice")
            return bundle

    def cached_voices(self) -> tuple[VoiceBundle, ...]:
        if not self.voices_dir.exists():
            return ()
        bundles: list[VoiceBundle] = []
        try:
            catalog = self._load_catalog()
        except AssetError:
            catalog = {"voices": {}}
        for directory in sorted(path for path in self.voices_dir.iterdir() if path.is_dir()):
            entry = catalog.get("voices", {}).get(directory.name)
            metadata = self._metadata(entry, catalog) if isinstance(entry, Mapping) else None
            try:
                bundles.append(VoiceBundle.from_directory(directory, metadata=metadata))
            except Exception:
                continue
        return tuple(bundles)

    def remove_voice(self, voice: str) -> None:
        import shutil

        target = self._safe_voice_path(self.get_voice_metadata(voice).id)
        if target.exists():
            shutil.rmtree(target)

    def prune(self) -> tuple[Path, ...]:
        removed: list[Path] = []
        if not self.voices_dir.exists():
            return ()
        known = {metadata.id for metadata in self.list_voices()}
        for path in self.voices_dir.iterdir():
            if path.is_dir() and path.name not in known:
                import shutil

                shutil.rmtree(path)
                removed.append(path)
        return tuple(removed)

    def clear(self, *, voices: bool = False) -> None:
        import shutil

        if voices:
            shutil.rmtree(self.voices_dir, ignore_errors=True)
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def cache_info(self) -> CacheInfo:
        return CacheInfo(
            directory=self.cache_dir,
            catalog_path=self.catalog_path,
            cached_voices=tuple(
                bundle.voice_id for bundle in self.cached_voices() if bundle.voice_id is not None
            ),
            catalog_cached=self.catalog_path.exists(),
        )


def list_voices(
    *,
    language: str | None = None,
    quality: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh: bool = False,
) -> tuple[VoiceMetadata, ...]:
    """List catalog voices using the standard asset manager."""

    return VoiceAssetManager(cache_dir, offline=offline).list_voices(
        language=language, quality=quality, refresh=refresh
    )


def list_cached_voices(
    *,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
) -> tuple[VoiceBundle, ...]:
    """List valid voice bundles currently present in the cache."""

    return VoiceAssetManager(cache_dir, offline=offline).cached_voices()
