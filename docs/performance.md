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

## Calibration catalog provenance

PiperSynth 0.2.0 carries forward the packaged 2,704-record catalog unchanged. It uses the `pipersynth-count-1-to-10-v1` corpus, BS.1770 measurements, and a `-24.0 LUFS` reference. The catalog's `generated_with` metadata records the original toolchain:

- `audiosig` 0.1.4
- `onnxvoice` 0.1.9
- `piperg2p` 0.1.5
- `pipersynth` 0.1.3.dev1+ge4f1c6860
- `spokenform` 0.4.5
- `utterplan` 0.1.3

These measurements were not recalibrated under the 0.2.0 dependency stack. Nine catalog identities remain unavailable and absent from the catalog:

- `piper:ar_JO-kareem-low:low:speaker-0`
- `piper:ar_JO-kareem-medium:medium:speaker-0`
- `piper:he_IL-saspeech-medium:medium:speaker-0`
- `piper:ja_JP-hi_fi_captain-medium:medium:speaker-0`
- `piper:ja_JP-hi_fi_captain-medium:medium:speaker-1`
- `piper:lt_LT-reginute1-medium:medium:speaker-0`
- `piper:th_TH-tsync2-medium:medium:speaker-0`
- `piper:zh_CN-chaowen-medium:medium:speaker-0`
- `piper:zh_CN-xiao_ya-medium:medium:speaker-0`

The retained `piper:en_US-libritts_r-medium:medium:speaker-761` measurement has a gain of `-12.195012852417848 dB`, below the normal `-12.0 dB` minimum. `benchmarks/data/voice_level_policy.json` records an exact-identity minimum equal to that measurement, with rationale. The global floor remains unchanged for every other identity. This exception makes future promotion reproducible without altering the measurement; revisit it when recalibrating the catalog. The packaged catalog itself remains unchanged for this release.

## Re-measuring calibration

The calibration benchmark accepts explicit prepared text and a matching Piper language. It measures the supplied text as-is and does not expand numbers. Review its measurement report before promoting data. Promotion writes a separate catalog and refuses to overwrite the packaged production catalog.

See `benchmarks/README.md` for the measurement and promotion workflow.
