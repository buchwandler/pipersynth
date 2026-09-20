#!/usr/bin/env python3
"""Inventory and optionally synthesize every Piper catalog voice identity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pipersynth import GenerationConfig, LoudnessConfig, PiperPipeline, VoiceAssetManager
from pipersynth.assets import VoiceMetadata
from pipersynth.audio import write_wav
from pipersynth.voice_level import VoiceCalibrationKey

try:
    from examples._output import artefact_dir
except ModuleNotFoundError:
    from _output import artefact_dir


@dataclass(frozen=True, slots=True)
class VoiceCatalogEntry:
    number: int
    model_id: str
    name: str
    locale: str
    language_family: str
    region: str | None
    quality: str
    num_speakers: int
    speaker_names: tuple[str, ...]
    aliases: tuple[str, ...]
    source_revision: str


@dataclass(frozen=True, slots=True)
class VoiceIdentityEntry:
    number: int
    model_id: str
    quality: str
    locale: str
    speaker_id: int
    speaker_name: str | None
    calibration_key: str


@dataclass(frozen=True, slots=True)
class SpokenformProbe:
    status: Literal["supported", "unsupported", "unavailable"]
    normalized_language: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class LanguageCoverage:
    locale: str
    base_language: str
    catalog_voice_count: int
    speaker_identity_count: int
    example_voice_ids: tuple[str, ...]
    spokenform_status: Literal["supported", "unsupported", "unavailable"]
    normalized_language: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ShowcaseCatalog:
    voices: tuple[VoiceCatalogEntry, ...]
    identities: tuple[VoiceIdentityEntry, ...]
    languages: tuple[LanguageCoverage, ...]


def _base_language(locale: str) -> str:
    return locale.replace("-", "_").split("_", 1)[0].lower()


def _speaker_map(metadata: VoiceMetadata) -> dict[int, str]:
    return {int(identifier): str(name) for name, identifier in metadata.speaker_id_map.items()}


def build_catalog(voices: Sequence[VoiceMetadata]) -> ShowcaseCatalog:
    """Build catalog rows and exact speaker identities without resolving assets."""

    ordered = sorted(voices, key=lambda item: (item.language_code, item.id, item.quality))
    catalog_rows: list[VoiceCatalogEntry] = []
    identity_rows: list[VoiceIdentityEntry] = []
    for voice_number, metadata in enumerate(ordered, 1):
        names_by_id = _speaker_map(metadata)
        speaker_names = tuple(
            names_by_id[speaker_id]
            for speaker_id in range(max(1, metadata.num_speakers))
            if speaker_id in names_by_id
        )
        catalog_rows.append(
            VoiceCatalogEntry(
                number=voice_number,
                model_id=metadata.id,
                name=metadata.name,
                locale=metadata.language_code,
                language_family=metadata.language_family,
                region=metadata.region,
                quality=metadata.quality,
                num_speakers=metadata.num_speakers,
                speaker_names=speaker_names,
                aliases=metadata.aliases,
                source_revision=metadata.source_revision,
            )
        )
        for speaker_id in range(max(1, metadata.num_speakers)):
            key = VoiceCalibrationKey("piper", metadata.id, metadata.quality, f"speaker-{speaker_id}")
            identity_rows.append(
                VoiceIdentityEntry(
                    number=len(identity_rows) + 1,
                    model_id=metadata.id,
                    quality=metadata.quality,
                    locale=metadata.language_code,
                    speaker_id=speaker_id,
                    speaker_name=names_by_id.get(speaker_id),
                    calibration_key=str(key),
                )
            )
    return ShowcaseCatalog(tuple(catalog_rows), tuple(identity_rows), ())


def probe_spokenform_locale(locale: str) -> SpokenformProbe:
    """Probe one exact Piper locale without making Spokenform a discovery dependency."""

    try:
        from spokenform import normalize_language
    except ImportError as error:
        return SpokenformProbe("unavailable", None, str(error))
    try:
        normalized = str(normalize_language(locale))
    except Exception as error:
        return SpokenformProbe("unsupported", None, str(error))
    return SpokenformProbe("supported", normalized, None)


def build_language_coverage(
    catalog: ShowcaseCatalog,
    probe: Callable[[str], SpokenformProbe] = probe_spokenform_locale,
) -> tuple[LanguageCoverage, ...]:
    voices_by_locale: dict[str, list[VoiceCatalogEntry]] = defaultdict(list)
    identities_by_locale: dict[str, list[VoiceIdentityEntry]] = defaultdict(list)
    for voice in catalog.voices:
        voices_by_locale[voice.locale].append(voice)
    for identity in catalog.identities:
        identities_by_locale[identity.locale].append(identity)

    rows: list[LanguageCoverage] = []
    for locale in sorted(voices_by_locale):
        result = probe(locale)
        rows.append(
            LanguageCoverage(
                locale=locale,
                base_language=_base_language(locale),
                catalog_voice_count=len(voices_by_locale[locale]),
                speaker_identity_count=len(identities_by_locale[locale]),
                example_voice_ids=tuple(item.model_id for item in voices_by_locale[locale][:3]),
                spokenform_status=result.status,
                normalized_language=result.normalized_language,
                error=result.error,
            )
        )
    return tuple(rows)


def with_language_coverage(
    catalog: ShowcaseCatalog,
    probe: Callable[[str], SpokenformProbe] = probe_spokenform_locale,
) -> ShowcaseCatalog:
    return ShowcaseCatalog(catalog.voices, catalog.identities, build_language_coverage(catalog, probe))


def _voice_dict(item: VoiceCatalogEntry) -> dict[str, object]:
    return asdict(item) | {
        "speaker_names": list(item.speaker_names),
        "aliases": list(item.aliases),
    }


def _identity_dict(item: VoiceIdentityEntry) -> dict[str, object]:
    return asdict(item)


def _language_dict(item: LanguageCoverage) -> dict[str, object]:
    return asdict(item) | {
        "example_voice_ids": list(item.example_voice_ids),
        "spokenform": {
            "status": item.spokenform_status,
            "normalized_language": item.normalized_language,
            "error": item.error,
        },
    }


def inventory_payload(catalog: ShowcaseCatalog) -> dict[str, object]:
    return {
        "schema": 1,
        "catalog_voice_count": len(catalog.voices),
        "speaker_identity_count": len(catalog.identities),
        "voices": [_voice_dict(item) for item in catalog.voices],
        "identities": [_identity_dict(item) for item in catalog.identities],
        "languages": [_language_dict(item) for item in catalog.languages],
    }


def language_payload(catalog: ShowcaseCatalog) -> dict[str, object]:
    return {
        "schema": 1,
        "catalog_voice_count": len(catalog.voices),
        "speaker_identity_count": len(catalog.identities),
        "languages": [_language_dict(item) for item in catalog.languages],
    }


def missing_language_payload(catalog: ShowcaseCatalog) -> dict[str, object]:
    missing = [
        _language_dict(item)
        for item in catalog.languages
        if item.spokenform_status != "supported"
    ]
    return {
        "schema": 1,
        "catalog_voice_count": len(catalog.voices),
        "speaker_identity_count": len(catalog.identities),
        "locales": missing,
        "languages": missing,
    }


def write_inventory_outputs(catalog: ShowcaseCatalog, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "inventory": output_dir / "all_voices_inventory.json",
        "languages": output_dir / "all_languages.json",
        "missing": output_dir / "spokenform_missing_languages.json",
        "missing_text": output_dir / "spokenform_missing_languages.txt",
    }
    outputs["inventory"].write_text(
        json.dumps(inventory_payload(catalog), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs["languages"].write_text(
        json.dumps(language_payload(catalog), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs["missing"].write_text(
        json.dumps(missing_language_payload(catalog), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing_locales = sorted(
        item.locale for item in catalog.languages if item.spokenform_status != "supported"
    )
    outputs["missing_text"].write_text("\n".join(missing_locales) + ("\n" if missing_locales else ""), encoding="utf-8")
    return outputs


def format_language_table(languages: Sequence[LanguageCoverage]) -> str:
    lines = ["Piper locales", "-------------"]
    lines.extend(item.locale for item in languages)
    return "\n".join(lines)


def format_voice_table(voices: Sequence[VoiceCatalogEntry]) -> str:
    lines = [
        "No.  Voice/model                    Locale   Quality  Speakers  Revision",
        "---  ------------------------------  -------  -------  --------  --------",
    ]
    lines.extend(
        f"{item.number:>3}  {item.model_id:<30}  {item.locale:<7}  {item.quality:<7}  "
        f"{item.num_speakers:>8}  {item.source_revision}"
        for item in voices
    )
    return "\n".join(lines)


def format_identity_table(identities: Sequence[VoiceIdentityEntry]) -> str:
    lines = ["No.  Calibration identity                                      Name", "---  ---------------------------------------------------------  ----"]
    lines.extend(
        f"{item.number:>3}  {item.calibration_key:<57}  {item.speaker_name or ''}"
        for item in identities
    )
    return "\n".join(lines)


def print_summary(catalog: ShowcaseCatalog, cache_dir: Path) -> None:
    supported = sum(item.spokenform_status == "supported" for item in catalog.languages)
    unsupported = len(catalog.languages) - supported
    print("PiperSynth all-voices showcase")
    print()
    print(f"Catalog cache: {cache_dir}")
    print(f"Catalog voices: {len(catalog.voices)}")
    print(f"Expanded speaker identities: {len(catalog.identities)}")
    print(f"Distinct Piper locales: {len(catalog.languages)}")
    print()
    print(format_language_table(catalog.languages))
    print()
    print("Spokenform coverage")
    print("-------------------")
    print(f"supported:   {supported}")
    print(f"unsupported: {unsupported}")
    print()
    print("Unsupported Piper locales")
    print("-------------------------")
    for item in catalog.languages:
        if item.spokenform_status != "supported":
            print(
                f"{item.locale}    base={item.base_language}    "
                f"catalog_voices={item.catalog_voice_count}    "
                f"speaker_identities={item.speaker_identity_count}"
            )
    print()
    print(format_voice_table(catalog.voices))


def _selected_metadata(
    voices: Sequence[VoiceMetadata],
    *,
    voice: str | None,
    language: str | None,
    quality: str | None,
    max_voices: int | None,
) -> tuple[VoiceMetadata, ...]:
    selected = list(voices)
    if voice is not None:
        selected = [item for item in selected if item.id == voice or voice in item.aliases]
    if language is not None:
        selected = [item for item in selected if item.language_code == language]
    if quality is not None:
        selected = [item for item in selected if item.quality == quality]
    selected.sort(key=lambda item: (item.language_code, item.id, item.quality))
    if max_voices is not None:
        selected = selected[:max_voices]
    return tuple(selected)


def _synthesis_text(locale: str) -> str:
    from benchmarks.voice_loudness import resolve_count_stimulus

    return resolve_count_stimulus(locale).spoken_text


def synthesize_catalog(
    catalog: ShowcaseCatalog,
    *,
    output_dir: Path,
    cache_dir: Path | None,
    offline: bool,
    refresh_catalog: bool,
    voice_leveling: Literal["off", "calibrated"],
    print_levels: bool,
) -> None:
    supported_locales = {
        item.locale for item in catalog.languages if item.spokenform_status == "supported"
    }
    identities = [item for item in catalog.identities if item.locale in supported_locales]
    by_model: dict[str, list[VoiceIdentityEntry]] = defaultdict(list)
    for identity in identities:
        by_model[identity.model_id].append(identity)

    audio_by_rate: dict[int, list[np.ndarray]] = defaultdict(list)
    manifest: list[dict[str, object]] = []
    plans_written = False
    for model_id in sorted(by_model):
        model_identities = by_model[model_id]
        first = model_identities[0]
        with PiperPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            generation=GenerationConfig(speaker=first.speaker_id),
            loudness=LoudnessConfig(voice_leveling=voice_leveling),
            text_preparation="identity",
            language=first.locale,
        ) as pipeline:
            for identity in model_identities:
                index = len(manifest) + 1
                print(f"[{index}/{len(identities)}] {identity.calibration_key}", flush=True)
                result = pipeline.run(_synthesis_text(identity.locale), speaker=identity.speaker_id)
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                start_frame = sum(part.size for part in audio_by_rate[result.sample_rate])
                audio_by_rate[result.sample_rate].append(audio)
                end_frame = start_frame + audio.size
                manifest.append(
                    {
                        "model_id": identity.model_id,
                        "quality": identity.quality,
                        "locale": identity.locale,
                        "speaker_id": identity.speaker_id,
                        "speaker_name": identity.speaker_name,
                        "calibration_key": identity.calibration_key,
                        "sample_rate": result.sample_rate,
                        "output_file": f"all_voices_{result.sample_rate}.wav",
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "duration_seconds": audio.size / result.sample_rate,
                        "announcement": _synthesis_text(identity.locale),
                    }
                )
                if not plans_written:
                    plan = pipeline.plan(_synthesis_text(identity.locale), unit="sentence")
                    plan.save(output_dir / "all_voices.utterplan.json")
                    plans_written = True

    for sample_rate, parts in sorted(audio_by_rate.items()):
        write_wav(output_dir / f"all_voices_{sample_rate}.wav", np.concatenate(parts), sample_rate)
    (output_dir / "all_voices_manifest.json").write_text(
        json.dumps({"schema": 1, "identities": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if print_levels:
        print("Voice leveling: " + voice_leveling)
    print(f"Rendered {len(manifest)} supported speaker identities.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--show-identities", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--voice")
    parser.add_argument("--quality")
    parser.add_argument("--speaker", type=int)
    parser.add_argument("--max-voices", type=int)
    parser.add_argument("--skip-unsupported", action="store_true")
    parser.add_argument("--voice-leveling", choices=("off", "calibrated"), default="off")
    parser.add_argument("--print-levels", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = VoiceAssetManager(args.cache_dir, offline=args.offline)
    metadata = manager.list_voices(refresh=args.refresh_catalog)
    selected = _selected_metadata(
        metadata,
        voice=args.voice,
        language=args.language,
        quality=args.quality,
        max_voices=args.max_voices,
    )
    catalog = build_catalog(selected)
    if args.speaker is not None:
        catalog = ShowcaseCatalog(
            catalog.voices,
            tuple(item for item in catalog.identities if item.speaker_id == args.speaker),
            (),
        )
    catalog = with_language_coverage(catalog)
    output_dir = artefact_dir()
    write_inventory_outputs(catalog, output_dir)
    print_summary(catalog, manager.cache_info().directory)
    if args.show_identities:
        print()
        print(format_identity_table(catalog.identities))
    sys.stdout.flush()

    if args.list_only:
        return 0
    unsupported = [item for item in catalog.languages if item.spokenform_status != "supported"]
    if unsupported and not args.skip_unsupported:
        print(
            f"Cannot synthesize the complete catalog: {len(unsupported)} Piper locales are not accepted by Spokenform."
        )
        print(f"See {output_dir / 'spokenform_missing_languages.txt'}.")
        print("Use --list-only for inventory or --skip-unsupported for a partial showcase.")
        return 2
    if unsupported:
        print(f"Partial showcase: skipping {len(unsupported)} unsupported Piper locales.")
    synthesize_catalog(
        catalog,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh_catalog=args.refresh_catalog,
        voice_leveling=args.voice_leveling,
        print_levels=args.print_levels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
