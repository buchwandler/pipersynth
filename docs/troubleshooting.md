# Troubleshooting

## OnnxVoice or ONNX Runtime is missing

Install `pipersynth[cpu]` or `pipersynth[gpu]`. Importing `pipersynth` itself does not require ONNX Runtime. PiperSynth delegates runtime dependency errors to its own `OptionalDependencyError` with the relevant extra.

## Managed voice or catalog errors

Managed voice references are normalized to `piper:<voice-id>` and resolved by OnnxVoice. Use `offline=True` only when the catalog and installation are already cached. Catalog, integrity, and lock failures are translated to PiperSynth asset errors while preserving the original exception as the cause.

## Model contract errors

A Piper-compatible model exposes `input`, `input_lengths`, and `scales`. Multi-speaker models also expose `sid`. The matching `.onnx.json` file must exist beside the model unless `config_path` is supplied. OnnxVoice owns tensor construction and reports contract errors through PiperSynth model errors.

## Invalid speaker errors

Single-speaker voices accept no explicit speaker or ID `0`. Multi-speaker IDs must be in range, and names must exist in `speaker_id_map`. No speaker is silently changed.

## Non-finite or unexpected audio

Model output must be a finite singleton-wrapped waveform at the native sample rate declared by the Piper config. Ambiguous multi-channel or multi-batch output is rejected. PCM conversion clips only finite normalized values and never writes NaN or infinity.

## Text preparation

Identity mode is the default. `spokenform` requires an explicit language. It is optional and runs before `piperg2p`; raw `[[...]]` phoneme blocks are protected.
