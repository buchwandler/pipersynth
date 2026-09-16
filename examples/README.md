# PiperSynth examples

The maintained examples use catalog voices and the UtterPlan workflow:

1. Resolve a voice with `PiperPipeline.from_pretrained(...)`.
2. Build an explicit plan with `pipeline.plan(...)`.
3. Save the `*.utterplan.json` artifact.
4. Render that existing plan and save a WAV file.

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

Direct phoneme mode is intentionally excluded from this suite because
`is_phonemes=True` bypasses UtterPlan. The library convenience APIs remain supported;
the examples use explicit planning to make the semantic boundary visible.
