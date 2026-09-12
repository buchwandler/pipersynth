from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigFileNotFoundError, ModelFileNotFoundError, OptionalDependencyError


@dataclass(frozen=True, slots=True)
class VoiceBundle:
    """Local Piper voice artifacts, including model-specific licensing metadata."""

    model_path: Path
    config_path: Path
    model_card: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> VoiceBundle:
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
        return cls(model, config, card)

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
