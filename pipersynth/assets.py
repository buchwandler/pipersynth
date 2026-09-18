from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._onnxvoice import installation_to_voice_info, normalize_piper_ref
from .errors import ConfigFileNotFoundError, ModelFileNotFoundError


@dataclass(frozen=True, slots=True)
class VoiceMetadata:
    """Read-only Piper catalog metadata supplied by OnnxVoice."""

    id: str
    name: str
    language_code: str
    language_family: str
    region: str | None
    quality: str
    num_speakers: int
    speaker_id_map: Mapping[str, int]
    aliases: tuple[str, ...]
    source_revision: str

    @classmethod
    def from_catalog_entry(
        cls, entry: Mapping[str, Any], *, source_revision: str = "unknown"
    ) -> VoiceMetadata:
        language = entry.get("language")
        if not isinstance(language, Mapping):
            raise ValueError("catalog voice language must be an object")
        region = language.get("region")
        return cls(
            id=str(entry["id"]),
            name=str(entry.get("name") or entry["id"]),
            language_code=str(language.get("code") or ""),
            language_family=str(language.get("family") or ""),
            region=str(region) if region is not None else None,
            quality=str(entry.get("quality") or ""),
            num_speakers=int(entry.get("num_speakers") or 1),
            speaker_id_map={
                str(name): int(identifier)
                for name, identifier in dict(entry.get("speaker_id_map", {})).items()
            },
            aliases=tuple(str(alias) for alias in entry.get("aliases", ())),
            source_revision=source_revision,
        )

    @classmethod
    def from_installation(cls, installation: Any) -> VoiceMetadata:
        metadata = dict(getattr(installation, "metadata", {}) or {})
        language = metadata.get("language") or {}
        if not isinstance(language, Mapping):
            language = {}
        return cls(
            id=str(getattr(installation, "id", "")),
            name=str(metadata.get("name") or getattr(installation, "id", "")),
            language_code=str(language.get("code") or ""),
            language_family=str(language.get("family") or ""),
            region=(str(language["region"]) if language.get("region") is not None else None),
            quality=str(metadata.get("quality") or ""),
            num_speakers=int(metadata.get("num_speakers") or 1),
            speaker_id_map={
                str(name): int(identifier)
                for name, identifier in dict(metadata.get("speaker_id_map") or {}).items()
            },
            aliases=tuple(str(alias) for alias in metadata.get("aliases") or ()),
            source_revision=str(metadata.get("source_revision") or "unknown"),
        )


@dataclass(frozen=True, slots=True)
class VoiceBundle:
    """Compatibility view over an OnnxVoice installation."""

    model_path: Path
    config_path: Path
    model_card: Path | None = None
    metadata: VoiceMetadata | None = None
    installation: Any | None = None

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, metadata: VoiceMetadata | None = None
    ) -> VoiceBundle:
        root = Path(directory)
        models = tuple(root.glob("*.onnx"))
        if len(models) != 1:
            raise ModelFileNotFoundError(f"Expected exactly one .onnx model in {root}")
        model = models[0]
        config = Path(f"{model}.json")
        card = root / "MODEL_CARD"
        if not config.exists():
            raise ConfigFileNotFoundError(f"Voice config file does not exist: {config}")
        return cls(model, config, card if card.exists() else None, metadata)

    @classmethod
    def from_installation(cls, installation: Any) -> VoiceBundle:
        resolved = installation_to_voice_info(installation)
        return cls(
            resolved.model_path,
            resolved.config_path,
            resolved.model_card_path,
            VoiceMetadata.from_installation(installation),
            installation,
        )

    @property
    def voice_id(self) -> str | None:
        return self.metadata.id if self.metadata is not None else None

    @property
    def directory(self) -> Path:
        return self.model_path.parent

    @property
    def model_card_text(self) -> str:
        return self.model_card.read_text(encoding="utf-8") if self.model_card else ""


def load_catalog_voice(
    voice_id: str,
    *,
    cache_dir: str | Path,
    catalog_path: str | Path,
    download: bool = False,
) -> VoiceBundle:
    """Resolve a Piper installation through OnnxVoice's catalog and store."""

    import onnxvoice

    manager = onnxvoice.OnnxVoice(
        cache_dir=cache_dir,
        catalog_sources={"piper": str(Path(catalog_path))},
    )
    ref = normalize_piper_ref(voice_id)
    if download:
        installation = manager.install(ref)
    else:
        installation = manager.resolve(ref)
    return VoiceBundle.from_installation(installation)
