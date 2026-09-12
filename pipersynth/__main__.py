from __future__ import annotations

import argparse

from .config import GenerationConfig, PipelineConfig
from .pipeline import PiperPipeline


def _speaker(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
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
    args = parser.parse_args()

    noise_w_scale = args.noise_w_scale
    if args.noise_w is not None:
        if noise_w_scale is not None:
            parser.error("--noise-w and --noise-w-scale cannot both be set")
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


if __name__ == "__main__":
    main()
