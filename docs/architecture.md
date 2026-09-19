# Architecture

PiperSynth is the application-facing Piper engine. Its dependencies have distinct ownership:

1. `UtterPlan` owns document parsing, SSMD, written-to-spoken preparation, language runs, semantic directives, pause resolution, markers, and render units.
2. `piperg2p` owns Piper voice JSON parsing, frontend dispatch, phonemization, raw phoneme blocks, lexicons, and voice-specific ID encoding.
3. `OnnxVoice` owns Piper model catalogs, installation caches, artifact verification, provider selection, ONNX Runtime sessions, Piper graph tensors, and raw native-rate inference results.
4. `AudioCompose` owns generic audio sources, explicit silence, timeline composition, resampling, anchors, final clipping, and AudioJob persistence.
5. PiperSynth owns application policy: speaker names and defaults, synthesis controls, UtterPlan adaptation, compatibility postprocessing, result metadata, diagnostics, and the public `PiperPipeline` and `PiperVoice` APIs.

The normal batch chain is:

```text
source text
  -> UtterPlan
  -> PiperSynth plan adapter
  -> piperg2p phonemes and Piper IDs
  -> OnnxVoice PiperAdapter
  -> raw float32 model audio
  -> PiperSynth compatibility interpretation
  -> AudioJob
  -> AudioCompose Composer
  -> PiperSynth AudioResult
```

`AudioCompose` receives only generic `AudioClip`, `Silence`, and `AudioJob` values. Piper-specific details remain opaque JSON-safe metadata. It does not import PiperSynth, OnnxVoice, piperg2p, or UtterPlan, and it does not branch on producer names.

## Runtime and asset lifecycle

`PiperVoice.load()` validates local model and config paths, parses `VoiceConfig`, creates `PiperFrontend`, and opens the local model through `onnxvoice.open_local(system="piper", ...)`. `PiperVoice.from_pretrained()` and `PiperPipeline.from_pretrained()` normalize unqualified IDs to `piper:<id>`, install through `onnxvoice.OnnxVoice`, and open the resulting installation. PiperSynth asset classes are compatibility views over those installations, not independent catalogs or caches.

OnnxVoice returns raw model audio. PiperSynth resolves speaker IDs and scalar synthesis controls before calling `infer()`, validates the model sample rate against `VoiceConfig`, and keeps the existing normalize and volume compatibility step. Raw inference result objects and NumPy auxiliary tensors never enter an AudioJob.

## AudioJob and composition

`PiperPipeline.to_audio_job(plan, **render_overrides)` is producer-only. It creates one `AudioClip` per rendered segment using the segment ID, explicit positive `Silence` items for resolved pauses, stable boundary anchors, JSON-safe provenance, and an explicit compatibility output policy. It does not call `Composer`.

`render_plan()` uses the same private render context, calls `Composer.compose()` exactly once, and takes the final waveform and sample rate from `CompositionResult`. It rebuilds PiperSynth markers, warnings, diagnostics, and optional retained unit chunks from the composition context. Streaming APIs remain a separate specialized path because complete-document composition and streaming have different lifetime and loudness semantics.

## Lifecycle

Use `with PiperVoice.load(...)` and `with PiperPipeline(...)` where possible. `close()` is idempotent and closes prepared plans, the planner, and owned OnnxVoice runtimes. Injected voice factories and planners remain caller-managed dependencies.

The package is clean-room independent from Piper's GPL runtime. Compatibility is expressed through the external model protocol, voice configuration, and public behavior, not copied implementation code.

## Offline voice loudness calibration

Voice leveling is an offline catalog-data flow, not a synthesis-time measurement:

```text
complete OnnxVoice Piper catalog
  -> expand every numeric speaker
  -> count 1..10 stimulus per locale
  -> three repeats with leveling off and normalize_audio=True
  -> audiosig BS.1770 integrated LUFS and true peak
  -> median/MAD and headroom-safe gain
  -> reviewable schema-2 report
  -> strict promotion to runtime schema 1
```

Managed pipelines retain the `VoiceBundle.installation`, so runtime lookup uses the canonical installation ID, metadata quality, and resolved numeric speaker. Local path loads intentionally have no catalog identity. The exact key is `piper:model-id:quality:speaker-N`; aliases and display names are never used as calibration identities.

Runtime processing is `raw inference -> optional legacy peak normalization -> static catalog or override gain -> user/segment volume -> finite validation -> one final clamp`. The static gain is cheap and deterministic. Missing identity or missing record produces a zero-gain diagnostic rather than a fuzzy substitution.

`target_lufs` is a separate complete-output `AudioCompose` policy. It is valid for composed batch output and rejected by true streaming APIs because normalizing each streamed unit would not normalize the finished document. The benchmark uses `voice_leveling="off"`, `target_lufs=None`, `normalize_audio=True`, and volume `1.0` so its measurements match the shipped processing baseline.

The benchmark enumerates the complete current `VoiceAssetManager.list_voices(refresh=True)` inventory and expands `range(max(1, num_speakers))`. Filtered or offline runs are development evidence only and cannot pass the full-coverage promotion gate. Production promotion validates provenance, policy, repeat count, finite values, MAD, exact identities, and coverage, recomputes true-peak-safe gains, strips benchmark-only fields, and refuses the packaged output path.
