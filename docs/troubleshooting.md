# Troubleshooting

## ONNX Runtime is missing

Install `pipersynth[cpu]` or `pipersynth[gpu]`. Importing `pipersynth` itself does not require ONNX Runtime.

## Model contract errors

A Piper-compatible model exposes `input`, `input_lengths`, and `scales`. Multi-speaker models also expose `sid`. The matching `.onnx.json` file must exist beside the model unless `config_path` is supplied.

## Invalid speaker errors

Single-speaker voices accept no explicit speaker or ID `0`. Multi-speaker IDs must be in range, and names must exist in `speaker_id_map`. No speaker is silently changed.

## Non-finite or unexpected audio

Model output must be a finite singleton-wrapped waveform. Ambiguous multi-channel or multi-batch output is rejected. PCM conversion clips only finite normalized values and never writes NaN or infinity.

## Text preparation

Identity mode is the default. `spokenform` requires an explicit language. It is optional and runs before `piperg2p`; raw `[[...]]` phoneme blocks are protected.
