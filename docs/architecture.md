# Architecture

PiperSynth has three boundaries:

1. `piperg2p` owns Piper voice JSON parsing, frontend dispatch, phonemization, raw phoneme blocks, lexicons, and voice-specific ID encoding.
2. PiperSynth owns ONNX Runtime sessions, provider selection, tensor construction, speaker validation, waveform postprocessing, PCM/WAV output, lifecycle, diagnostics, and unit streaming.
3. Optional adapters own written-to-spoken preparation and voice catalog provisioning.

The runtime chain is:

```text
source text -> optional preparation -> piperg2p -> phonemes and IDs
            -> PiperSynth ONNX session -> float32 audio -> PCM16/WAV/chunks
```

`PiperVoice` is the model-specific low-level runtime. `PiperPipeline` adds reusable configuration, preparation, final results, and sentence streaming. Sessions are created lazily by pipelines and reused across calls. Audio is not cached implicitly.

The package is clean-room independent from Piper's GPL runtime. Compatibility is expressed through the external model protocol, voice configuration, and public behavior, not copied implementation code.

## Lifecycle

Use `with PiperVoice.load(...)` and `with PiperPipeline(...)` where possible. `close()` is idempotent. Injected frontends are not closed by PiperSynth; frontends created by `PiperVoice.load` are owned and closed by that voice.

Sentence units come from `piperg2p.PhonemizeResult.sentences`. Paragraph mapping and source offsets are intentionally not fabricated when reliable mapping is unavailable.
