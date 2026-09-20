# PiperSynth examples

The maintained examples use catalog voices and the UtterPlan workflow:

1. Resolve a voice with `PiperPipeline.from_pretrained(...)`.
2. Build an explicit plan with `pipeline.plan(...)`.
3. Save the `*.utterplan.json` artifact.
4. Render that existing plan and save a WAV file.
5. Optionally convert the plan to an `AudioJob` and replay it through AudioCompose.

No example requires a manually downloaded `.onnx` file. Voice bundles are resolved
and cached by PiperSynth. Generated plans and WAV files are written below
`example-artefacts/`, or below the directory named by
`PIPERSYNTH_EXAMPLE_OUTPUT_DIR`.

## Commands

```bash
# Run one example.
python examples/basic.py

# Run maintained examples in isolated output directories.
python examples/run_all.py

# Include optional examples, which may download another voice.
python examples/run_all.py --include-optional

# Use cached voices only after an online run.
PIPERSYNTH_OFFLINE=1 python examples/run_all.py

# Override the normal English catalog voice.
PIPERSYNTH_EXAMPLE_VOICE=en_US-lessac-high python examples/basic.py
```


## All Piper voices and languages

Print the current Piper catalog without downloading voice models:

```bash
python examples/all_voices.py --list-only
```

The command prints every catalog voice, the expanded speaker-identity count, exact Piper locale codes, and current Spokenform coverage. It writes `all_voices_inventory.json`, `all_languages.json`, `spokenform_missing_languages.json`, and the one-locale-per-line `spokenform_missing_languages.txt` handoff below `example-artefacts/`.

A catalog voice is a model entry. A speaker identity is one numeric speaker within that model, so one multi-speaker catalog voice can produce many calibration identities. The full showcase is resource-heavy and is not run by default. Select it explicitly with:

```bash
python examples/run_all.py --include-resource-heavy
python examples/all_voices.py
```

Use `--skip-unsupported` only for a clearly marked partial showcase. The all-voices output groups WAV files by each model's native sample rate rather than silently resampling them.


## Loudness benchmark preflight

The loudness benchmark resolves every distinct Piper locale and count stimulus before creating the first synthesis pipeline. Unsupported locales are reported together and represented as identity-level failures. Use `python benchmarks/voice_loudness.py --list-stimuli` to inspect this preflight without inference.

Direct phoneme mode is intentionally excluded from this suite because
`is_phonemes=True` bypasses UtterPlan. The library convenience APIs remain supported;
the examples use explicit planning to make the semantic boundary visible.
