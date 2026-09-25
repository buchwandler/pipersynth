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

## Atomic prepared requests

PiperSynth never chooses a new text boundary. Readio or another caller prepares and splits text before sending each atomic request. `PiperVoice.synthesize()` and `synthesize_text()` pass the exact request text to PiperG2P and perform at most one acoustic inference. Frontend sentence groups are joined as phoneme IDs for that single inference; PiperSynth does not call Phrasplit or split on punctuation.

```python
from pipersynth import PiperVoice, SynthesisRequest

prepared_text = (
    "The first already-prepared sentence. "
    "The next sentence remains part of this same atomic request."
)

request = SynthesisRequest(
    id="paragraph-001",
    text=prepared_text,
    language="en-us",
)

with PiperVoice.from_pretrained("en_US-lessac-medium") as voice:
    result = voice.synthesize(request)
    result.save_wav("paragraph.wav")
```

If the complete request exceeds a capacity reported by the model or frontend, PiperSynth raises `SynthesisInputTooLongError`. It does not retry with smaller pieces. Readio decides the next boundary.
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

Use `SynthesisRequest` for a request with an explicit identity, speaker, pronunciation override, or linguistic token context. Token offsets refer to the exact prepared string, and all supplied token fields, including `morph`, are forwarded to PiperG2P.

```python
from pipersynth import (
    LinguisticToken,
    PiperVoice,
    PronunciationOverride,
    SynthesisRequest,
)

request = SynthesisRequest(
    id="line-001",
    text="I read the book yesterday.",
    language="en-us",
    pronunciation_overrides=(PronunciationOverride(2, 6, phonemes="ɹɛd"),),
    tokens=(
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
    result = voice.synthesize(request)
```

`SynthesisSegment` remains accepted as a compatibility request type. Its `annotations` become `SynthesisRequest.tokens`; it follows the same atomic behavior. `synthesize_text()` is also strict. None of these APIs accepts `chunking`.

## Piper frontend and low-level IDs

PiperG2P may represent one atomic request as several frontend sentence groups. PiperSynth joins their phoneme IDs and performs one model inference for the complete request. It does not expose those groups as independently rendered chunks.

`PiperVoice.synthesize_ids()` remains available for callers that already have Piper phoneme IDs. ID validation, speaker selection, acoustic controls, voice calibration, and waveform postprocessing still apply.

## Audio and calibration

Results contain mono finite `float32` audio at the model's native sample rate. `SynthesisResult.save_wav()` writes mono 16-bit PCM. Conversion clips to the supported PCM range.

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
