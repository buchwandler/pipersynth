# PiperSynth

PiperSynth is an application-facing Python synthesis library for Piper-compatible ONNX voices. It uses `piperg2p` for voice configuration and phonemization, `OnnxVoice` for model assets and ONNX execution, and `AudioCompose` for generic audio composition and AudioJob persistence. It does not depend on the upstream Piper runtime or `piper-tts`.

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

On first use OnnxVoice fetches the Piper catalog and installs the selected model, matching config, and any model card into its shared local store. Later calls reuse that installation. The convenience call creates a fresh pipeline and closes it before returning.
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

## Planning and rendering

`PiperPipeline.plan()` compiles text into an immutable `UtterancePlan`. The UtterPlan planner owns document parsing, Spokenform, SSMD, language runs, semantic units, markers, and resolved pauses. Rendering an existing plan never replans it, so the same plan can be rendered repeatedly with different acoustic overrides:

```python
with PiperPipeline.from_pretrained("en_US-lessac-medium") as pipe:
    plan = pipe.plan("One. Two.", unit="sentence")
    plan.save("speech.utterplan.json")
    normal = pipe.render_plan(plan, length_scale=1.0)
    fast = pipe.render_plan(plan, length_scale=0.9)
```

## AudioJob production and replay

An existing plan can be converted to a generic, persisted AudioJob without composing it in PiperSynth:

```python
job = pipe.to_audio_job(plan)
manifest = job.save("speech.audiojob")
```

`AudioClip` IDs preserve UtterPlan segment IDs. Resolved semantic pauses are explicit `Silence` items, and the job includes an explicit compatibility output policy. Replay is producer-neutral:

```python
from audiocompose import AudioJob, Composer

job = AudioJob.load("speech.audiojob/audiojob.json")
composition = Composer().compose(job)
```

The normal `render_plan()` API builds and composes this job exactly once, then adapts the composed waveform back to `AudioResult`. Streaming APIs remain a separate batch-independent path.

## Runnable examples

The maintained examples use catalog voices and require no manual model download.
They explicitly create and persist an `UtterancePlan` before rendering it. Generated plans
and WAV files are written below `example-artefacts/`. See [`examples/README.md`](examples/README.md).

```bash
python examples/basic.py
python examples/run_all.py
```

Use `is_phonemes=True` only for direct Piper phoneme input. It bypasses UtterPlan and does not attach a semantic plan to the result.

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

Set `ONNXVOICE_CACHE_DIR` or pass `cache_dir=` explicitly to control the OnnxVoice store. `PIPERSYNTH_CACHE_DIR` remains accepted as a PiperSynth compatibility alias. Set `PIPERSYNTH_OFFLINE=1` for process-wide offline operation. Explicit `offline=` arguments take precedence.

OnnxVoice owns installed artifacts, manifests, checksums, and locks. PiperSynth's `VoiceAssetManager` and `VoiceBundle` are compatibility views over that store. Voice licenses apply to the downloaded model and are not part of the PiperSynth Apache-2.0 license.

The Python API provides catalog and cache operations:

```python
from pipersynth import VoiceAssetManager

manager = VoiceAssetManager()

# List voices
for voice in manager.list_voices(language="en", quality="medium"):
    print(voice.id, voice.name)

# Get voice metadata
metadata = manager.get_voice_metadata("en_US-lessac-medium")
print(metadata.id, metadata.name, metadata.language_code, metadata.quality)

# Download/resolve a voice
bundle = manager.resolve_voice("en_US-lessac-medium")
print(bundle.directory)
print(bundle.model_card_text)

# Cache operations
print(manager.cache_info())
for cached in manager.cached_voices():
    print(cached.directory)

# Destructive operations (use with caution)
# manager.remove_voice("en_US-lessac-medium")
# manager.prune()
# manager.clear(voices=True)
```

## Optional features

The UtterPlan dependency provides Spokenform and SSMD planning. Install `pipersynth[playback]` for `AudioResult.play()` and streaming playback, or `pipersynth[gpu]` for the OnnxVoice GPU provider. Catalog support is provided by OnnxVoice and is also available as `pipersynth[catalog]`.

The core API supports sentence and paragraph units through UtterPlan, resolved semantic pauses, plan save/load, and PCM iteration through `iter_pcm()`. It does not claim generic voice blending, approximate word timings, hidden language detection, model conversion, training, quantization, or HTTP serving.

See [`docs/architecture.md`](docs/architecture.md), [`docs/providers.md`](docs/providers.md), [`docs/troubleshooting.md`](docs/troubleshooting.md), and [`examples/download_and_synthesize.py`](examples/download_and_synthesize.py).
