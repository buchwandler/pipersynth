# Piper voice calibration benchmark

The benchmark measures only explicitly prepared speech through `PiperVoice`. It does not invoke Spokenform, expand numbers, or synthesize a document plan. Choose a text stimulus and `--language` that match the model's PiperG2P profile.

List the exact model and speaker identities that would be measured:

```bash
python benchmarks/voice_level_benchmark.py \
  --language en_US --quality medium --max-voices 1 --list-only
```

Run a small, reproducible measurement first:

```bash
python benchmarks/voice_level_benchmark.py \
  --language en_US \
  --text "Please leave the package beside the front door." \
  --quality medium \
  --max-voices 1
```

The stimulus is synthesized three times per model speaker by default. A model is loaded once for all its speakers. The JSON report contains the exact prepared stimulus, catalog identity, individual BS.1770 measurements, median, median absolute deviation, static gain candidate, software versions, and per-identity coverage. Failed model or inference phases are recorded in the report. The command returns nonzero when any measurement fails.

By default, output is written to `benchmarks/output/voice_level_calibration/measurements.json`. Generated output is ignored by git. Use `--voice`, `--quality`, `--speaker`, `--cache-dir`, `--offline`, and `--refresh-catalog` to constrain or configure a run. The benchmark never modifies the packaged calibration data.

## Candidate promotion

Promotion validates the report and writes a separate candidate catalog:

```bash
python benchmarks/promote_voice_calibration.py \
  benchmarks/output/voice_level_calibration/measurements.json
```

Incomplete identity coverage requires `--allow-partial`. High-variability results are excluded unless explicitly included with `--include-high-variability`. An existing output requires `--force`. The packaged catalog path is rejected as an output target. Review the report and candidate before any separate, deliberate update to packaged runtime calibration data.

The policy, including repeat count, reference loudness, allowed gain range, and variability threshold, is in `benchmarks/data/voice_level_policy.json`.
