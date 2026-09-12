from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigFileNotFoundError, ModelFileNotFoundError, OptionalDependencyError


@dataclass(frozen=True, slots=True)
class VoiceMetadata:
    """Typed catalog metadata for one Piper voice."""

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
    def from_catalog_entry(cls, entry: Mapping[str, Any], *, source_revision: str) -> VoiceMetadata:
        language = entry.get("language")
        if not isinstance(language, Mapping):
            raise ValueError("catalog voice language must be an object")
        region = language.get("region")
        return cls(
            id=str(entry["id"]),
            name=str(entry["name"]),
            language_code=str(language["code"]),
            language_family=str(language["family"]),
            region=str(region) if region is not None else None,
            quality=str(entry["quality"]),
            num_speakers=int(entry["num_speakers"]),
            speaker_id_map={str(name): int(identifier) for name, identifier in dict(entry.get("speaker_id_map", {})).items()},
            aliases=tuple(str(alias) for alias in entry.get("aliases", ())),
            source_revision=source_revision,
        )


@dataclass(frozen=True, slots=True)
class VoiceBundle:
    """Local Piper voice artifacts, including model-specific licensing metadata."""

    model_path: Path
    config_path: Path
    model_card: Path
    metadata: VoiceMetadata | None = None

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
        if not card.exists():
            raise FileNotFoundError(f"Voice MODEL_CARD does not exist: {card}")
        return cls(model, config, card, metadata)

    @property
    def voice_id(self) -> str | None:
        return self.metadata.id if self.metadata is not None else None

    @property
    def directory(self) -> Path:
        return self.model_path.parent

    @property
    def model_card_text(self) -> str:
        return self.model_card.read_text(encoding="utf-8")


def load_catalog_voice(
    voice_id: str,
    *,
    cache_dir: str | Path,
    catalog_path: str | Path,
    download: bool = False,
 ) -> VoiceBundle:
    """Resolve a voice from a local catalog, downloading only when explicitly requested."""

    try:
        from piper_voice_catalog import download_voice, get_voice, load_catalog
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "catalog integration requires piper-onnx-voices. Install pipersynth[catalog]."
        ) from exc
    catalog = load_catalog(Path(catalog_path))
    voice = get_voice(catalog, voice_id)
    target = Path(cache_dir) / str(voice["id"])
    if download:
        download_voice(voice, target)
    return VoiceBundle.from_directory(target)
