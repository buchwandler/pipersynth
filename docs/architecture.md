# Architecture

PiperSynth is a Piper synthesis engine. It accepts one prepared speech request, uses PiperG2P to produce Piper sentence groups and phoneme IDs, calls OnnxVoice for model inference, applies engine-local audio processing, and returns one independent `RenderedSegment`.

## Ownership

| Responsibility                                                                             | Owner                         |
| ------------------------------------------------------------------------------------------ | ----------------------------- |
| Document parsing, SSMD, written-to-spoken preparation, logical voice roles                 | Application or document layer |
| Piper frontend, phonemization, sentence groups, Piper IDs                                  | PiperG2P                      |
| Voice catalogs, asset installation, provider selection, ONNX sessions, inference           | OnnxVoice                     |
| Piper speaker resolution, acoustic scales, request-local joining, static voice calibration | PiperSynth                    |
| Timeline pauses, markers, resampling, mixing, final mastering                              | Caller or AudioCompose        |

PiperSynth has no runtime dependency on Utterplan, SSMD, or AudioCompose. Those systems may construct `SynthesisSegment` requests and compose returned audio outside this package.

## Request lifecycle

1. The caller prepares speakable text and creates a `SynthesisSegment` or calls `synthesize_text()`.
2. PiperSynth validates the active model language and resolves a speaker name or ID within that model.
3. PiperSynth converts source-aligned pronunciation overrides and token annotations to PiperG2P types. It preserves offsets and fields such as `morph`.
4. PiperG2P returns ordered sentence groups. PiperSynth infers every nonempty ID group separately.
5. PiperSynth validates each native-rate waveform, applies normalization, static voice calibration, and explicit `output_gain`, then joins the chunks directly without inserted silence.
6. The request returns one `RenderedSegment`. Runtime timing and output-tensor summaries remain diagnostics, not word alignment.

Each request has independent audio. PiperSynth does not create a global timeline, markers, document pauses, or fabricated word timings.

## Voice lifecycle

`PiperVoice.from_pretrained()` installs or reuses an OnnxVoice-managed Piper voice. `PiperVoice.from_local()` opens a local model through OnnxVoice and uses the adjacent `<model>.json` config unless another config path is supplied. Context-manager exit closes the runtime and any frontend created by PiperSynth.

A `PiperVoice` remains bound to one Piper model. A different model requires another voice instance. A different speaker within a multi-speaker model is selected on `SynthesisSegment.speaker` or `synthesize_text(speaker=...)`.

## Audio policy

In-memory audio is mono finite `float32` at the Piper model's sample rate. WAV output uses mono signed 16-bit PCM. Piper acoustic controls are explicit `SynthesisConfig` fields. `output_gain` and static voice-level calibration are engine-local controls. Complete-output loudness and true-peak mastering are not part of PiperSynth.
