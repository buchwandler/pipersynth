#!/usr/bin/env python3
"""List Piper catalog voices and optionally synthesize one prepared sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipersynth import PiperVoice, VoiceAssetManager

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language")
    parser.add_argument("--quality")
    parser.add_argument("--voice", help="synthesize with one exact catalog voice ID")
    parser.add_argument("--speaker", help="numeric speaker ID or model speaker name")
    parser.add_argument("--text", default="A prepared speech sample.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = VoiceAssetManager(args.cache_dir, offline=args.offline)
    voices = manager.list_voices(
        language=args.language, quality=args.quality, refresh=args.refresh_catalog
    )
    inventory = [
        {
            "id": voice.id,
            "name": voice.name,
            "language": voice.language_code,
            "quality": voice.quality,
            "num_speakers": voice.num_speakers,
            "speaker_id_map": dict(voice.speaker_id_map),
        }
        for voice in voices
    ]
    output = artefact_path("all_voices.json")
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for voice in voices:
        print(f"{voice.id}\t{voice.language_code}\t{voice.quality}\t{voice.num_speakers}")
    print(f"Catalog entries: {len(voices)}")
    print(f"Inventory: {output}")

    if args.voice:
        metadata = manager.get_voice_metadata(args.voice)
        speaker: int | str | None = args.speaker
        if isinstance(speaker, str) and speaker.isdecimal():
            speaker = int(speaker)
        with PiperVoice.from_pretrained(
            metadata.id, cache_dir=args.cache_dir, offline=args.offline
        ) as voice:
            result = voice.synthesize_text(
                args.text,
                language=metadata.language_code,
                speaker=speaker,
            )
        wav_path = artefact_path(f"{metadata.id}.wav")
        result.save_wav(wav_path)
        print(f"WAV: {wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
