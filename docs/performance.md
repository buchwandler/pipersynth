# Performance and voice calibration

## Reuse a voice

Opening a Piper runtime is more expensive than an individual inference call. Reuse one `PiperVoice` for repeated requests:

```python
from pipersynth import PiperVoice
with PiperVoice.from_pretrained("en_US-lessac-medium") as voice:
    first = voice.synthesize_text("First prepared request.", language="en-us")
    second = voice.synthesize_text("Second prepared request.", language="en-us")
```

`iter_chunks()` first derives prepared-text parts with Phrasplit's exact offsets in regex mode, then phonemizes and infers each part incrementally. PiperG2P still owns its sentence groups, so one text part can yield multiple rendered chunks. `synthesize()` joins them in source order without adding silence.

`TextChunkingConfig` defaults to sentence splitting and accepts `max_chars` for long or run-on text. Chunk metadata identifies the exact `[char_start, char_end)` slice of the original prepared request, and the result summary reports the splitter diagnostics and number of text parts. Overrides, annotations, and raw phoneme blocks protect their complete spans; merging can make a part longer than the requested character limit. Use `mode="none"` to skip PiperSynth splitting while retaining PiperG2P sentence groups.

These are synthesis-engine chunks, not semantic pauses or document timeline segments. Use caller-side streaming or AudioCompose when a larger application needs document-level composition.

## Static voice-level calibration

PiperSynth may apply one deterministic calibration gain for an exact model, quality, and numeric speaker identity. The calibration lookup key is `piper:model-id:quality:speaker-N`. Single-speaker voices use speaker 0. Anonymous local voices have no inferred catalog identity; an explicit `VoiceLevelConfig.gain_db` remains available when a caller supplies a reviewed gain.

Calibration is applied after optional request peak normalization and before explicit `output_gain` and the final safety clamp. It is a fixed model/speaker correction. It does not measure each request at runtime and does not guarantee a target loudness for arbitrary text.

The packaged calibration catalog is `pipersynth/data/voice_level_calibration.json`. It records measurements and static gains. Final LUFS and true-peak mastering belong to the caller's final-output layer.

## Re-measuring calibration

The calibration benchmark accepts explicit prepared text and a matching Piper language. It measures the supplied text as-is and does not expand numbers. Review its measurement report before promoting data. Promotion writes a separate catalog and refuses to overwrite the packaged production catalog.

See `benchmarks/README.md` for the measurement and promotion workflow.
