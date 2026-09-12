# PiperSynth

PiperSynth is an independent Apache-2.0 Python runtime for Piper-compatible ONNX voice models. It uses `piperg2p` for voice configuration, text-to-phoneme conversion, and phoneme IDs, and owns ONNX inference, audio conversion, WAV writing, lifecycle, and streaming. It does not depend on the upstream Piper runtime or `piper-tts`.

## Quick start

Install the CPU runtime and catalog support:

```bash
pip install "pipersynth[cpu]"
```

Generate a WAV from a catalog voice:

```python
from pipersynth import synthesize_to_wav

synthesize_to_wav(
    "Hello, this sentence was generated with PiperSynth.",
    "hello.wav",
    voice="en_US-lessac-medium",
)
```

On first use PiperSynth fetches the voice catalog and downloads the selected model, matching config, and `MODEL_CARD` into its local cache. Later calls reuse the cached assets. The convenience call creates a fresh pipeline and closes it before returning.

For repeated synthesis, reuse one pipeline and one ONNX session:

```python
from pipersynth import PiperPipeline

with PiperPipeline.from_pretrained("en_US-lessac-medium") as pipe:
    pipe("One.").save_wav("one.wav")
    pipe("Two.").save_wav("two.wav")
```

Use cached assets only with `offline=True`:

```python
with PiperPipeline.from_pretrained("en_US-lessac-medium", offline=True) as pipe:
    pipe("This uses cached assets only.").save_wav("offline.wav")
```

## Local models

Existing explicit local model usage remains network-free:

```python
from pipersynth import PiperPipeline, PipelineConfig

with PiperPipeline(PipelineConfig(model_path="voice.onnx")) as pipe:
    pipe("No network is used here.").save_wav("local.wav")
```

`PiperVoice.load()` and `PiperPipeline(PipelineConfig(...))` never resolve the catalog or download assets. Use `PiperVoice.from_pretrained()` or `PiperPipeline.from_pretrained()` when managed catalog resources are desired.

## Voice discovery and cache

```python
from pipersynth import VoiceAssetManager, list_voices

for voice in list_voices(language="en", quality="medium"):
    print(voice.id, voice.name)

manager = VoiceAssetManager()
metadata = manager.get_voice_metadata("en_US-lessac-medium")
bundle = manager.resolve_voice("en_US-lessac-medium")
print(bundle.model_card_text)
```

Set `PIPERSYNTH_CACHE_DIR` to override the platform cache location, or pass `cache_dir=` explicitly. Set `PIPERSYNTH_OFFLINE=1` for process-wide offline operation. Explicit `offline=` arguments take precedence.

Each downloaded bundle contains the upstream `MODEL_CARD`. Voice licenses apply to the downloaded model and are not part of the PiperSynth Apache-2.0 license.

The CLI provides catalog and cache operations:

```bash
pipersynth voices list --language en --quality medium
pipersynth voices show en_US-lessac-medium
pipersynth voices download en_US-lessac-medium
pipersynth voices license en_US-lessac-medium
pipersynth voices path en_US-lessac-medium
pipersynth speak --voice en_US-lessac-medium "Hello from PiperSynth." -o hello.wav
pipersynth cache info
```

The original local-model command remains supported:

```bash
pipersynth voice.onnx "Hello world." -o hello.wav
```

## Optional features

Install `pipersynth[spokenform]` for written-text preparation, `pipersynth[playback]` for `AudioResult.play()` and streaming playback, or `pipersynth[gpu]` for GPU ONNX Runtime. Catalog support is included in the CPU and GPU extras and is also available as `pipersynth[catalog]`.

The core API supports sentence units and real paragraph grouping through `prepare_units(..., unit="paragraph")`, plus PCM iteration through `iter_pcm()`. It does not claim generic voice blending, approximate word timings, hidden language detection, full SSMD support, model conversion, training, quantization, or HTTP serving.

See [`docs/architecture.md`](docs/architecture.md), [`docs/providers.md`](docs/providers.md), [`docs/troubleshooting.md`](docs/troubleshooting.md), and [`examples/download_and_synthesize.py`](examples/download_and_synthesize.py).
