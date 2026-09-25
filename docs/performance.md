# Performance and voice calibration

## Reuse a voice

Opening a Piper runtime is more expensive than an individual inference call. Reuse one `PiperVoice` for repeated requests:

```python
from pipersynth import PiperVoice
with PiperVoice.from_pretrained("en_US-lessac-medium") as voice:
    first = voice.synthesize_text("First prepared request.", language="en-us")
    second = voice.synthesize_text("Second prepared request.", language="en-us")
```

A single atomic request is phonemized once and sent through one acoustic inference. PiperG2P may represent frontend sentence groups internally, but PiperSynth joins their phoneme IDs and does not expose or infer each group independently.

PiperSynth does not split long or run-on requests. If the model or frontend reports a known capacity, PiperSynth raises `SynthesisInputTooLongError` with the available capacity details. No universal character or phoneme limit is assumed. The caller, usually Readio, chooses any smaller request boundaries before synthesis.

These are engine requests, not semantic pauses or document timeline segments. Use caller-side streaming or AudioCompose when a larger application needs document-level composition.

## Static voice-level calibration

PiperSynth may apply one deterministic calibration gain for an exact model, quality, and numeric speaker identity. The calibration lookup key is `piper:model-id:quality:speaker-N`. Single-speaker voices use speaker 0. Anonymous local voices have no inferred catalog identity; an explicit `VoiceLevelConfig.gain_db` remains available when a caller supplies a reviewed gain.

Calibration is applied after optional request peak normalization and before explicit `output_gain` and the final safety clamp. It is a fixed model/speaker correction. It does not measure each request at runtime and does not guarantee a target loudness for arbitrary text.

The packaged calibration catalog is `pipersynth/data/voice_level_calibration.json`. It records measurements and static gains. Final LUFS and true-peak mastering belong to the caller's final-output layer.

## Re-measuring calibration

The calibration benchmark accepts explicit prepared text and a matching Piper language. It measures the supplied text as-is and does not expand numbers. Review its measurement report before promoting data. Promotion writes a separate catalog and refuses to overwrite the packaged production catalog.

See `benchmarks/README.md` for the measurement and promotion workflow.
