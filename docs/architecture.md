# Architecture

PiperSynth has three boundaries:

1. `UtterPlan` owns document parsing, SSMD, written-to-spoken preparation, language runs, semantic directives, pause resolution, markers, and render units.
2. `piperg2p` owns Piper voice JSON parsing, frontend dispatch, phonemization, raw phoneme blocks, lexicons, and voice-specific ID encoding.
3. PiperSynth owns ONNX Runtime sessions, provider selection, tensor construction, speaker validation, waveform postprocessing, PCM/WAV output, lifecycle, diagnostics, and audio unit streaming.

The runtime chain is:

source text -> UtterPlan -> PiperG2P -> phonemes and IDs
-> PiperSynth ONNX session -> float32 audio -> PCM16/WAV/chunks

```

`PiperVoice` is the model-specific low-level runtime. `PiperPipeline` combines reusable UtterPlan configuration, plan compilation, Piper rendering, final results, and PlanUnit streaming. Sessions are created lazily by pipelines and reused across calls. Audio is not cached implicitly.

## Lifecycle

Use `with PiperVoice.load(...)` and `with PiperPipeline(...)` where possible. `close()` is idempotent and closes prepared plans, the planner, and owned voices. Injected voice factories and planners remain caller-managed dependencies.

Plan units come from `UtterPlan.units`, and planned rendering uses only resolved plan pauses. Marker character positions are always exposed; sample offsets are returned only for known segment or unit boundaries.

The package is clean-room independent from Piper's GPL runtime. Compatibility is expressed through the external model protocol, voice configuration, and public behavior, not copied implementation code.
```
