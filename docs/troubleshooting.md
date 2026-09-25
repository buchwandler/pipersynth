# Troubleshooting

## Missing ONNX Runtime

Install one runtime extra:

```bash
pip install "pipersynth[cpu]"
# or
pip install "pipersynth[gpu]"
```

Catalog and model asset access are provided by OnnxVoice. Use `offline=True` or `PIPERSYNTH_OFFLINE=1` when only cached assets should be used.

## Language does not match the active model

A `PiperVoice` is bound to one model. Pass a language compatible with that model's PiperG2P profile. PiperSynth does not switch acoustic models based on request text. For a different model, open a different `PiperVoice`.

## Unknown or invalid speaker

Speaker names and numeric IDs must belong to the active Piper model. `None` selects the configured default for a multi-speaker voice. A document role such as `guest` is not a Piper speaker unless that exact name exists in the model's speaker map.

## Pronunciation override or annotation offsets fail

Offsets use Python half-open ranges into the exact prepared string stored in `SynthesisSegment.text`. Do not reuse offsets from an unprepared source document. If `LinguisticToken.text` is supplied, it must equal the source slice.

## Text is pronounced differently than expected

PiperSynth accepts prepared, speakable text. It does not expand numbers, dates, abbreviations, SSMD, or written-to-spoken semantics. Perform that preparation in the caller, then pass the resulting text and any source-aligned pronunciation context.

## WAV output

Rendered audio is mono finite `float32` at the active model's sample rate. `RenderedSegment.save_wav()` writes mono 16-bit PCM. Use `output_gain` only for explicit engine-local gain. Final loudness, peak, and timeline policies belong to the caller.
