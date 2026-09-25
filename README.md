# PiperSynth

PiperSynth is a standalone synthesis engine for Piper-compatible ONNX voices. Callers provide prepared, speakable text; PiperG2P owns phonemization and Piper ID generation, OnnxVoice handles model inference, and PiperSynth applies engine-local acoustic and audio policy. Document parsing, SSMD, written-to-spoken preparation, semantic pauses, markers, timeline composition, and final mastering belong to other layers.

## Install

Install PiperSynth with the CPU runtime:

```bash
pip install "pipersynth[cpu]"
```

For GPU inference, install `pipersynth[gpu]`. OnnxVoice manages catalog lookup, model downloads, caching, provider selection, and ONNX Runtime sessions.

## Synthesize prepared text

```python
from pipersynth import PiperVoice

with PiperVoice.from_pretrained("en_US-lessac-medium") as voice:
    result = voice.synthesize_text(
        "Hello, this text is already prepared for speech.",
        language="en-us",
    )
    result.save_wav("hello.wav")
```

`prepared_text` is ordinary speakable text, not phoneme IDs. PiperSynth does not expand numbers, dates, abbreviations, or other written forms. Perform that semantic preparation before calling the engine.

## Long prepared text

Sentence chunking is the default. `max_chars` can add request-local splits for long or run-on text. PiperSynth uses Phrasplit in regex mode and preserves exact ranges into the prepared string. Boundaries that cross a pronunciation override, token annotation, or raw `[[...]]` phoneme block are merged, so a protected span is never clipped. A merged part can therefore exceed `max_chars`.

```python
from pipersynth import PiperVoice, TextChunkingConfig

prepared_text = (
    "The first prepared paragraph describes the field observations and the "
    "conditions recorded during the morning survey.\n\n"
    "A second paragraph continues the account without asking PiperSynth to "
    "interpret or prepare a source document."
)

with PiperVoice.from_pretrained("en_US-lessac-medium") as voice:
    result = voice.synthesize_text(
        prepared_text,
        language="en-us",
        chunking=TextChunkingConfig(max_chars=100),
    )

for chunk in result.chunks:
    source = chunk.metadata["text_chunk"]
    print(source["char_start"], source["char_end"])
```

Every chunk's `text_chunk` metadata contains `char_start`, `char_end`, and the split mode. Sentence-mode chunks also include the regex backend and Phrasplit split ID. The request summary is available in `result.metadata["text_chunking"]`. `TextChunkingConfig(mode="none")` disables only PiperSynth's pre-segmentation; PiperG2P may still return its normal sentence groups.
For repeated requests, reuse one voice:

```python
from pipersynth import PiperVoice, SynthesisConfig

with PiperVoice.from_pretrained("en_US-lessac-medium", offline=True) as voice:
    result = voice.synthesize_text(
        "A second prepared request.",
        language="en-us",
        config=SynthesisConfig(length_scale=0.9, output_gain=0.8),
    )
```

Use `PiperVoice.from_local("voice.onnx")` to open a local model. By default, PiperSynth reads its config from `voice.onnx.json`.

## Typed requests and linguistic context

Use `SynthesisSegment` when a request has an explicit identity, speaker, pronunciation override, or external token annotation:

```python
from pipersynth import (
    LinguisticToken,
    PiperVoice,
    PronunciationOverride,
    SynthesisSegment,
)

segment = SynthesisSegment(
    id="line-001",
    text="I read the book yesterday.",
    language="en-us",
    pronunciation_overrides=(PronunciationOverride(2, 6, phonemes="ɹɛd"),),
    annotations=(
        LinguisticToken(
            start=2,
            end=6,
            text="read",
            pos="VERB",
            lemma="read",
            morph="Tense=Past",
        ),
    ),
)

with PiperVoice.from_pretrained("en_US-lessac-medium") as voice:
    result = voice.synthesize(segment)
```

Offsets use Python half-open ranges into the exact prepared string. PiperSynth forwards annotation fields, including `morph`, to PiperG2P. Speaker names and numeric IDs identify speakers within the active Piper model, not document roles.

## Streaming and low-level IDs

`PiperVoice.iter_chunks()` applies the configured text pre-segmentation, then yields each nonempty PiperG2P sentence group as soon as it is inferred. `PiperVoice.synthesize()` joins those chunks in source order without adding silence. Each chunk exposes its source range in `chunk.metadata["text_chunk"]`; use `TextChunkingConfig(mode="none")` to bypass only PiperSynth's text pre-segmentation.

`PiperVoice.synthesize_ids()` remains available for callers that already have Piper phoneme IDs. ID validation, speaker selection, acoustic controls, voice calibration, and waveform postprocessing still apply.

## Audio and calibration

Results contain mono finite `float32` audio at the model's native sample rate. `RenderedSegment.save_wav()` writes mono 16-bit PCM. Conversion clips to the supported PCM range.

`SynthesisConfig` contains Piper controls (`length_scale`, `noise_scale`, and `noise_w_scale`), optional peak normalization, explicit engine-local `output_gain`, and static `voice_level` calibration. PiperSynth does not perform final LUFS or true-peak mastering. Use AudioCompose or another output layer when producing a document, chapter, or mixed timeline.

## Voice discovery and providers

```python
from pipersynth import VoiceAssetManager, list_voices

for item in list_voices(language="en", quality="medium"):
    print(item.id, item.name)

manager = VoiceAssetManager(offline=True)
print(manager.get_voice_metadata("en_US-lessac-medium"))
```

The default inference provider is `CPUExecutionProvider`. Pass `providers` and `provider_options` to `PiperVoice.from_pretrained()` or `PiperVoice.from_local()` to select runtime providers. See [provider configuration](docs/providers.md).

## Examples and development

Runnable examples are under [`examples/`](examples/). They synthesize prepared text with `PiperVoice`; generated files are written below `example-artefacts/` by default.

Run validation with:

```bash
python -m pytest
python -m ruff check pipersynth tests examples benchmarks
python -m mypy pipersynth tests/typecheck/engine_api.py
```
