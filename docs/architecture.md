# Architecture

PiperSynth is a Piper synthesis engine. It accepts one prepared speech request, uses PiperG2P to produce Piper sentence groups and phoneme IDs, calls OnnxVoice for model inference, applies engine-local audio processing, and returns one independent `RenderedSegment`.

## Ownership

| Responsibility                                                                                                    | Owner                       |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Document parsing, SSMD, written-to-spoken preparation, logical voice roles                                        | Caller or application layer |
| Piper phonemization, sentence groups, and Piper IDs                                                               | PiperG2P                    |
| Voice catalogs, asset installation, provider selection, ONNX sessions, and inference                              | OnnxVoice                   |
| Optional prepared-text splitting, source offsets, speaker resolution, chunk joining, and static voice calibration | PiperSynth                  |
| Timeline pauses, markers, resampling, mixing, and final mastering                                                 | Caller or AudioCompose      |

PiperSynth has no runtime dependency on Utterplan, SSMD, or AudioCompose. Those systems may construct `SynthesisSegment` requests and compose returned audio outside this package.

## Request lifecycle

1. The caller prepares speakable text and creates a `SynthesisSegment` or calls `synthesize_text()`.
2. `TextChunkingConfig` defaults to sentence pre-segmentation. PiperSynth lazily uses Phrasplit's exact offsets in regex mode, optionally with `max_chars`.
3. PiperSynth merges any candidate boundary that crosses an override, token annotation, or raw `[[...]]` block, then remaps protected offsets into each exact source slice. `mode="none"` bypasses this pre-segmentation and its Phrasplit import.
4. PiperSynth validates the active model language and speaker, then calls the configured PiperG2P path for each text part. PiperG2P continues to own phonemization, Piper IDs, and its normal sentence groups.
5. PiperSynth infers each nonempty G2P group, validates native-rate audio, applies normalization, static voice calibration, and explicit `output_gain`, then joins chunks in source order without inserted silence.
6. The request returns one `RenderedSegment` containing the complete prepared text, per-chunk source ranges, chunking summary, warnings, and frontend diagnostics. Runtime timing and output-tensor summaries remain diagnostics, not word alignment.

Each request has independent audio. PiperSynth does not create a global timeline, markers, document pauses, or fabricated word timings.

## Voice lifecycle

`PiperVoice.from_pretrained()` installs or reuses an OnnxVoice-managed Piper voice. `PiperVoice.from_local()` opens a local model through OnnxVoice and uses the adjacent `<model>.json` config unless another config path is supplied. Context-manager exit closes the OnnxVoice runtime.

A `PiperVoice` remains bound to one Piper model. A different model requires another voice instance. A different speaker within a multi-speaker model is selected on `SynthesisSegment.speaker` or `synthesize_text(speaker=...)`.

## Audio policy

In-memory audio is mono finite `float32` at the Piper model's sample rate. WAV output uses mono signed 16-bit PCM. Piper acoustic controls are explicit `SynthesisConfig` fields. `output_gain` and static voice-level calibration are engine-local controls. Complete-output loudness and true-peak mastering are not part of PiperSynth.
