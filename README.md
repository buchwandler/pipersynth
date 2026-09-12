# PiperSynth

PiperSynth is an independent Apache-2.0 Python runtime for Piper-compatible ONNX voice models. It uses `piperg2p` for voice configuration, text-to-phoneme conversion, and phoneme IDs, and owns the ONNX session, audio, lifecycle, and streaming layers. It does not depend on or import the Piper runtime or `piper-tts`.

## Install

The core package keeps ONNX Runtime optional:

```bash
pip install "pipersynth[cpu]"
```

Use `pipersynth[gpu]` for the GPU runtime, or install the package without either extra when injecting a test session. Optional written-to-spoken preparation and catalog support are available with `pipersynth[spokenform]` and `pipersynth[catalog]`.

## Basic use

```python
from pipersynth import PiperPipeline, PipelineConfig

with PiperPipeline(PipelineConfig(model_path="voice.onnx")) as pipe:
    result = pipe("Hello world.")
    result.save_wav("hello.wav")
```

`PiperPipeline` reuses one voice and ONNX session for later calls. Results retain both `source_text` and `prepared_text`.

## Low-level chunks

```python
from pipersynth import PiperVoice, SynthesisConfig

with PiperVoice.load("voice.onnx") as voice:
    for chunk in voice.synthesize("First sentence. Second sentence."):
        send_pcm(chunk.audio_int16_bytes)
```

The low-level API expects prepared/plain text. Text normalization is a pipeline concern.

## Expert controls

```python
from pipersynth import PiperVoice, SynthesisConfig

with PiperVoice.load(
    "voice.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
) as voice:
    audio = voice.synthesize_ids(
        [1, 0, 42, 0, 2],
        SynthesisConfig(length_scale=0.95, noise_scale=0.667, noise_w_scale=0.8),
    )
```

IDs must belong to the loaded voice's symbol map. Multi-speaker voices accept a validated `speaker_id`; configured names can be resolved with `voice.resolve_speaker_id(name)`.

## Written-to-spoken preparation

Identity preparation is the default. To opt in to `spokenform`, select a language explicitly:

```python
from pipersynth import PiperPipeline, PipelineConfig

pipe = PiperPipeline(PipelineConfig(
    model_path="de_DE-voice.onnx",
    text_preparation="spokenform",
    language="de",
))
result = pipe.run("Am 14.05.2026 sind es 2 kg.")
```

Raw `[[ phoneme ]]` blocks are protected from written-text normalization and remain owned by `piperg2p`.

## CLI

```bash
pipersynth voice.onnx "Hello world." -o hello.wav
pipersynth voice.onnx "Hello." --speaker alice --provider CPUExecutionProvider
```

The CLI supports config, speaker, acoustic scales, sentence silence, volume, normalization, provider, language, and `--prepare-text` controls. It is a thin client of the public pipeline API.

## Compatibility and scope

The core dependency range is `piperg2p>=0.2.0,<0.3`. Frontend capabilities are limited to those provided by the installed `piperg2p`; PiperSynth does not reimplement eSpeak or other G2P backends.

Voice artifacts are local-path-first: an ONNX model, matching `.onnx.json`, and model-specific `MODEL_CARD`. Catalog downloads are explicit and never occur during ordinary `PiperVoice.load`.

PiperSynth does not provide Piper alignment patching, word timings, SSMD, playback, HTTP serving, automatic language detection, model conversion, training, quantization, or voice blending.

See [`docs/architecture.md`](docs/architecture.md), [`docs/providers.md`](docs/providers.md), and [`docs/troubleshooting.md`](docs/troubleshooting.md) for details.
