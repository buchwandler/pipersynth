from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import onnxvoice
import pytest

from pipersynth.asset_manager import VoiceAssetManager
from pipersynth.errors import OfflineAssetError


class FakeInstallation:
    def __init__(self, root: Path) -> None:
        self.system = "piper"
        self.id = "en_US-test-medium"
        self.kind = "voice"
        self.path = root
        self.metadata = {
            "name": "Test voice",
            "language": {"code": "en_US", "family": "en", "region": "US"},
            "quality": "medium",
            "num_speakers": 1,
            "speaker_id_map": {},
            "source_revision": "a" * 40,
        }
        self._artifacts = {
            "model": SimpleNamespace(path=root / "test.onnx"),
            "config": SimpleNamespace(path=root / "test.onnx.json"),
            "model_card": SimpleNamespace(path=root / "MODEL_CARD"),
        }

    @property
    def ref(self) -> str:
        return f"{self.system}:{self.id}"

    def artifact(self, role: str):
        return self._artifacts[role]


class FakeCatalogItem:
    id = "en_US-test-medium"
    aliases = ("test",)
    metadata = {
        "name": "Test voice",
        "language": {"code": "en_US", "family": "en", "region": "US"},
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "source_revision": "a" * 40,
    }


class FakeOnnxVoice:
    installation: FakeInstallation
    calls: list[tuple[str, object]] = []

    def __init__(self, cache_dir=None, catalog_sources=None, offline=False):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(".")
        self.catalog_sources = catalog_sources
        self.offline = offline
        self.installation = FakeOnnxVoice.installation

    def install(self, ref, **kwargs):
        self.calls.append(("install", (ref, kwargs)))
        if self.offline:
            from onnxvoice.errors import OfflineError

            raise OfflineError("offline")
        progress = kwargs.get("progress")
        if progress:
            progress(SimpleNamespace(phase="download_started", ref=ref, message="download"))
            progress(SimpleNamespace(phase="download_completed", ref=ref, message="done"))
        return self.installation

    def list(self, system, **kwargs):
        self.calls.append(("list", (system, kwargs)))
        return [FakeCatalogItem()]

    def installed(self, system=None):
        return [self.installation]

    def resolve(self, ref):
        return self.installation

    def where(self, ref):
        return self.installation.path

    def remove(self, ref):
        self.calls.append(("remove", ref))


@pytest.fixture
def fake_onnxvoice(tmp_path: Path, monkeypatch):
    root = tmp_path / "voice"
    root.mkdir()
    (root / "test.onnx").write_bytes(b"model")
    (root / "test.onnx.json").write_text("{}")
    (root / "MODEL_CARD").write_text("license")
    FakeOnnxVoice.installation = FakeInstallation(root)
    FakeOnnxVoice.calls = []
    monkeypatch.setattr(onnxvoice, "OnnxVoice", FakeOnnxVoice)
    return root


def test_asset_manager_delegates_managed_install_and_adapts_progress(fake_onnxvoice, tmp_path):
    events = []
    manager = VoiceAssetManager(tmp_path, progress=events.append)
    bundle = manager.resolve_voice("test", refresh_catalog=True, force_download=True)

    assert bundle.voice_id == "en_US-test-medium"
    assert bundle.model_card_text == "license"
    assert FakeOnnxVoice.calls[0][0] == "install"
    ref, options = FakeOnnxVoice.calls[0][1]
    assert ref == "piper:test"
    assert options["refresh"] is True
    assert options["force"] is True
    assert [event.phase for event in events] == ["download-start", "download-complete"]


def test_asset_manager_lists_metadata_and_cached_installations(fake_onnxvoice, tmp_path):
    manager = VoiceAssetManager(tmp_path)
    metadata = manager.get_voice_metadata("test")

    assert metadata.id == "en_US-test-medium"
    assert metadata.language_code == "en_US"
    assert manager.cached_voices()[0].voice_id == metadata.id
    assert manager.is_voice_cached("test")
    assert manager.voice_cache_path("test").name == "voice"


def test_asset_manager_translates_onnxvoice_offline_error(fake_onnxvoice, tmp_path):
    manager = VoiceAssetManager(tmp_path, offline=True)
    with pytest.raises(OfflineAssetError):
        manager.resolve_voice("test")
