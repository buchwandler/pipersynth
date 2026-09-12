from __future__ import annotations

import argparse
from collections.abc import Sequence

from .asset_manager import VoiceAssetManager
from .config import GenerationConfig, PipelineConfig
from .convenience import synthesize_to_wav
from .pipeline import PiperPipeline


def _speaker(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with a Piper-compatible ONNX voice without the Piper runtime"
    )
    parser.add_argument("model", help="Path to .onnx voice model")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav")
    parser.add_argument("--config", default=None, help="Path to .onnx.json config")
    parser.add_argument("--speaker", type=_speaker, default=None)
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--noise-w-scale", type=float, default=None)
    parser.add_argument("--noise-w", type=float, default=None, help="Deprecated noise width alias")
    parser.add_argument("--sentence-silence", type=float, default=0.0)
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--provider", action="append", default=None, help="ONNX provider; repeat for priority")
    parser.add_argument("--language", default=None, help="Language for optional written-text preparation")
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare-text", action="store_true", help="Use spokenform preparation")
    preparation.add_argument("--no-prepare-text", action="store_true", help="Use identity preparation")
    parser.add_argument("--no-normalize", action="store_true")
    return parser


def _add_managed_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize speech with PiperSynth catalog voices")
    commands = parser.add_subparsers(dest="command", required=True)

    voices = commands.add_parser("voices")
    voice_commands = voices.add_subparsers(dest="voice_command", required=True)
    listing = voice_commands.add_parser("list")
    _add_managed_options(listing)
    listing.add_argument("--language")
    listing.add_argument("--quality")
    show = voice_commands.add_parser("show")
    _add_managed_options(show)
    show.add_argument("voice")
    for name in ("download", "path", "remove", "license"):
        command = voice_commands.add_parser(name)
        _add_managed_options(command)
        command.add_argument("voice")
    download = voice_commands.choices["download"]
    download.add_argument("--force-download", action="store_true")

    speak = commands.add_parser("speak")
    _add_managed_options(speak)
    speak.add_argument("--voice", required=True)
    speak.add_argument("text")
    speak.add_argument("-o", "--output", default="output.wav")
    speak.add_argument("--speaker", type=_speaker, default=None)
    speak.add_argument("--length-scale", type=float, default=None)
    speak.add_argument("--noise-scale", type=float, default=None)
    speak.add_argument("--noise-w-scale", type=float, default=None)
    speak.add_argument("--sentence-silence", type=float, default=0.0)
    speak.add_argument("--volume", type=float, default=1.0)
    speak.add_argument("--provider", action="append", default=None)
    speak.add_argument("--prepare-text", action="store_true")
    speak.add_argument("--language", default=None)
    speak.add_argument("--no-normalize", action="store_true")
    speak.add_argument("--force-download", action="store_true")

    cache = commands.add_parser("cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    for name in ("info", "list", "prune", "clear"):
        command = cache_commands.add_parser(name)
        command.add_argument("--cache-dir", default=None)
    cache_commands.choices["clear"].add_argument("--voices", action="store_true")
    return parser


def _run_legacy(args: argparse.Namespace) -> None:
    noise_w_scale = args.noise_w_scale
    if args.noise_w is not None:
        if noise_w_scale is not None:
            raise ValueError("--noise-w and --noise-w-scale cannot both be set")
        noise_w_scale = args.noise_w
    generation = GenerationConfig(
        speaker=args.speaker,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=not args.no_normalize,
        volume=args.volume,
        sentence_silence=args.sentence_silence,
    )
    config = PipelineConfig(
        model_path=args.model,
        config_path=args.config,
        generation=generation,
        providers=tuple(args.provider) if args.provider is not None else None,
        text_preparation="spokenform" if args.prepare_text else "identity",
        language=args.language,
    )
    with PiperPipeline(config) as pipeline:
        pipeline.run(args.text).save_wav(args.output)
    print(args.output)


def _run_managed(args: argparse.Namespace) -> None:
    offline = True if args.offline else None
    manager = VoiceAssetManager(args.cache_dir, offline=offline)
    if args.command == "voices":
        if args.voice_command == "list":
            for voice in manager.list_voices(
                language=args.language, quality=args.quality, refresh=args.refresh_catalog
            ):
                print(f"{voice.id}\t{voice.name}")
        elif args.voice_command == "show":
            voice = manager.get_voice_metadata(args.voice, refresh=args.refresh_catalog)
            print(f"id: {voice.id}")
            print(f"name: {voice.name}")
            print(f"language: {voice.language_code}")
            print(f"quality: {voice.quality}")
            print(f"speakers: {voice.num_speakers}")
            print(f"aliases: {', '.join(voice.aliases)}")
        elif args.voice_command == "download":
            bundle = manager.resolve_voice(
                args.voice,
                refresh_catalog=args.refresh_catalog,
                force_download=args.force_download,
            )
            print(bundle.directory)
        elif args.voice_command == "path":
            print(manager.voice_cache_path(args.voice))
        elif args.voice_command == "remove":
            manager.remove_voice(args.voice)
        elif args.voice_command == "license":
            print(manager.resolve_voice(args.voice, refresh_catalog=args.refresh_catalog).model_card_text)
        return
    if args.command == "speak":
        output = synthesize_to_wav(
            args.text,
            args.output,
            voice=args.voice,
            speaker=args.speaker,
            length_scale=args.length_scale,
            noise_scale=args.noise_scale,
            noise_w_scale=args.noise_w_scale,
            sentence_silence=args.sentence_silence,
            volume=args.volume,
            normalize_audio=not args.no_normalize,
            providers=args.provider,
            cache_dir=args.cache_dir,
            offline=offline,
            refresh_catalog=args.refresh_catalog,
            force_download=args.force_download,
            text_preparation="spokenform" if args.prepare_text else "identity",
            language=args.language,
        )
        print(output)
        return
    if args.cache_command == "info":
        info = manager.cache_info()
        print(f"directory: {info.directory}")
        print(f"catalog: {info.catalog_path}")
        print(f"cached voices: {', '.join(info.cached_voices)}")
    elif args.cache_command == "list":
        for bundle in manager.cached_voices():
            print(bundle.directory)
    elif args.cache_command == "prune":
        for path in manager.prune():
            print(path)
    elif args.cache_command == "clear":
        manager.clear(voices=args.voices)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]
    if not arguments or arguments[0] not in {"voices", "speak", "cache"}:
        _run_legacy(_legacy_parser().parse_args(arguments))
    else:
        _run_managed(_build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
