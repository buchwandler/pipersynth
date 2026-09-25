# Architecture

PiperSynth is a strict atomic-request engine for Piper-compatible ONNX voices. It accepts one caller-shaped request, calls PiperG2P once, joins frontend sentence representations into a single phoneme-ID sequence, runs at most one acoustic inference, applies engine-local audio policy, and returns one `SynthesisResult` or a typed error.

## Ownership

| Responsibility                                                                                | Owner                       |
| --------------------------------------------------------------------------------------------- | --------------------------- |
| Document parsing, SSMD, written-to-spoken preparation, and request boundaries                 | Caller or application layer |
| Piper phonemization, frontend sentence representation, and Piper IDs                          | PiperG2P                    |
| Voice catalogs, asset installation, provider selection, ONNX sessions, and inference          | OnnxVoice                   |
| Request validation, speaker resolution, static calibration, and engine-local audio processing | PiperSynth                  |
| Timeline pauses, markers, resampling, mixing, and final mastering                             | Caller or AudioCompose      |

PiperSynth has no runtime dependency on Utterplan, SSMD, or AudioCompose. Callers may construct `SynthesisRequest` values and compose the returned audio outside this package.

## Request lifecycle

1. The caller prepares speakable text, decides its boundaries, and creates a `SynthesisRequest` or uses the strict `synthesize_text()` wrapper.
2. PiperSynth validates text, language, speaker, token spans, pronunciation spans, model-language compatibility, and synthesis configuration.
3. PiperG2P receives the exact request text and caller-provided linguistic context once. PiperSynth does not call Phrasplit, split on punctuation, or rerun spaCy.
4. PiperSynth joins PiperG2P's frontend sentence representations into one phoneme-ID sequence. A known model capacity is checked before inference; an oversized request raises `SynthesisInputTooLongError`. PiperSynth never retries by splitting the request.
5. OnnxVoice performs one acoustic inference. PiperSynth validates finite mono audio, applies normalization, static voice calibration, and explicit `output_gain`, then returns `SynthesisResult`.
6. The result contains the request ID, exact text and language, sample rate, empty `word_timings`, warnings, and stable metadata. Internal rendered chunks are not part of the public result.

Each request returns independent audio. PiperSynth does not create a global timeline, markers, document pauses, or fabricated word timings. `supports_timestamps` is false.

## Voice lifecycle

`PiperVoice.from_pretrained()` installs or reuses an OnnxVoice-managed Piper voice. `PiperVoice.from_local()` opens a local model through OnnxVoice and uses the adjacent `<model>.json` config unless another config path is supplied. Context-manager exit closes the OnnxVoice runtime.

A `PiperVoice` remains bound to one Piper model. A different model requires another voice instance. A speaker is resolved within the active model. The result exposes the resolved speaker identity in metadata.

## Audio and identity policy

In-memory audio is mono finite `float32` at the Piper model's sample rate. WAV output uses mono signed 16-bit PCM. Piper acoustic controls are explicit `SynthesisConfig` fields. `output_gain` and static voice-level calibration are engine-local controls. Complete-output loudness and true-peak mastering are not part of PiperSynth.

`SynthesisResult.metadata["voice_level"]` reports calibration mode, whether a nonzero gain was applied, gain, source, calibration key, reason, and catalog revision. `synthesis_identity` and `synthesis_hash` include the model/catalog identity, resolved speaker and acoustic controls, calibration revision, PiperSynth version, and phoneme-affecting frontend options. Transient cache and progress settings are excluded.
