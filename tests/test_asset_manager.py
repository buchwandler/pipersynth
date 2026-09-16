from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pipersynth.asset_manager import VoiceAssetManager
from pipersynth.errors import AssetCacheError, AssetDownloadError, OfflineAssetError


def _fake_catalog() -> tuple[dict, dict[str, bytes]]:
    files = {"MODEL_CARD": b"license", "test.onnx": b"model", "test.onnx.json": b"config"}
    artifacts = {}
    for role, filename in (
        ("model_card", "MODEL_CARD"),
        ("model", "test.onnx"),
        ("config", "test.onnx.json"),
    ):
        data = files[filename]
        artifacts[role] = {
            "role": role,
            "path": filename,
            "filename": filename,
            "url": f"test://{filename}",
            "size": len(data),
            "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(),
        }
    entry = {
        "id": "en_US-test-medium",
        "name": "Test voice",
        "language": {"code": "en_US", "family": "en", "region": "US"},
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "aliases": ["test"],
        "artifacts": artifacts,
    }
    return {"source": {"revision": "a" * 40}, "voices": {entry["id"]: entry}}, files


def _manager(
    tmp_path: Path, *, offline: bool = False, progress=None
) -> tuple[VoiceAssetManager, dict, dict[str, bytes]]:
    catalog, files = _fake_catalog()
    manager = VoiceAssetManager(tmp_path, offline=offline, progress=progress)
    manager.catalog_path.parent.mkdir(parents=True)
    manager.catalog_path.write_text("ignored")

    def download(voice, target, *, overwrite=False):
        target.mkdir(parents=True, exist_ok=True)
        for filename, data in files.items():
            destination = target / filename
            if destination.exists() and not overwrite:
                continue
            destination.write_bytes(data)

    def get_voice(value, voice_id):
        if voice_id == "test":
            voice_id = "en_US-test-medium"
        return value["voices"][voice_id]

    def list_voices(value, *, language=None, quality=None):
        return list(value["voices"].values())

    manager._catalog_dependency = lambda: (
        ValueError,
        RuntimeError,
        download,
        lambda: catalog,
        get_voice,
        list_voices,
        lambda path: catalog,
    )
    return manager, catalog, files


def test_explicit_cache_path_and_alias_resolution(tmp_path: Path) -> None:
    events = []
    manager, _, _ = _manager(tmp_path, progress=events.append)
    bundle = manager.resolve_voice("test")
    assert bundle.voice_id == "en_US-test-medium"
    assert bundle.model_card_text == "license"
    assert bundle.directory == tmp_path / "voices" / "en_US-test-medium"
    assert [event.phase for event in events] == [
        "catalog-load",
        "voice-resolve",
        "download-start",
        "download-complete",
    ]


def test_cached_voice_is_reused_and_emits_cache_hit(tmp_path: Path) -> None:
    events = []
    manager, _, _ = _manager(tmp_path, progress=events.append)
    manager.resolve_voice("test")
    events.clear()
    manager.resolve_voice("en_US-test-medium")
    assert [event.phase for event in events] == ["catalog-load", "voice-resolve", "cache-hit"]
    assert manager.is_voice_cached("test")


def test_force_download_overwrites_even_valid_cache(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    manager.resolve_voice("test")
    calls = []
    original = manager._catalog_dependency
    dependencies = original()

    def download(*args, **kwargs):
        calls.append(True)
        return dependencies[2](*args, **kwargs)

    manager._catalog_dependency = lambda: (*dependencies[:2], download, *dependencies[3:])
    manager.resolve_voice("test", force_download=True)
    assert calls == [True]


def test_offline_missing_assets_raise(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path, offline=True)
    with pytest.raises(OfflineAssetError):
        manager.resolve_voice("test")


def test_corrupt_cache_requires_force_download(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    target = tmp_path / "voices" / "en_US-test-medium"
    target.mkdir(parents=True)
    (target / "MODEL_CARD").write_bytes(b"wrong")
    with pytest.raises(AssetCacheError):
        manager.resolve_voice("test")
    bundle = manager.resolve_voice("test", force_download=True)
    assert bundle.model_card_text == "license"


def test_download_errors_are_wrapped(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("download failed")

    manager._catalog_dependency = lambda: (
        ValueError,
        RuntimeError,
        fail,
        lambda: _fake_catalog()[0],
        lambda catalog, voice: (
            catalog["voices"]["en_US-test-medium"] if voice == "test" else catalog["voices"][voice]
        ),
        lambda catalog, **kwargs: list(catalog["voices"].values()),
        lambda path: _fake_catalog()[0],
    )
    with pytest.raises(AssetDownloadError):
        manager.resolve_voice("test")


def test_environment_cache_path_wins_when_no_explicit_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PIPERSYNTH_CACHE_DIR", str(tmp_path))
    assert VoiceAssetManager().cache_dir == tmp_path


def test_refresh_is_rejected_offline(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path, offline=True)
    with pytest.raises(OfflineAssetError):
        manager.list_voices(refresh=True)
