# PiperSynth examples

These examples demonstrate the PiperVoice synthesis-engine API. Text passed to PiperSynth must already be speakable. The examples do not parse documents or perform written-to-spoken semantic preparation.

Catalog examples use OnnxVoice-managed voices. The first run may download model assets; later runs reuse the cache. Output WAV files are written below `example-artefacts/` or the directory set by `PIPERSYNTH_EXAMPLE_OUTPUT_DIR`.

## Run examples

```bash
python examples/basic.py
python examples/run_all.py --list
python examples/run_all.py
```

Set `PIPERSYNTH_OFFLINE=1` to use cached assets only. Set `PIPERSYNTH_EXAMPLE_VOICE` to select another English catalog voice. `german.py` and `homographs.py` are optional examples and can be included with `--include-optional`.

## List catalog voices

```bash
python examples/all_voices.py --language en
```

This writes an inventory JSON file and does not download a voice. To synthesize one prepared sample, provide `--voice`:

```bash
python examples/all_voices.py \
  --voice en_US-lessac-medium \
  --text "This text is already prepared for speech."
```

For a multi-speaker model, pass a numeric ID or an actual model speaker name with `--speaker`. These are Piper model speakers, not document roles.

## Included examples

- `basic.py` synthesizes prepared English text.
- `download_and_synthesize.py` reports asset download progress.
- `german.py` uses a German Piper voice and explicit language.
- `homographs.py` forwards a source-aligned pronunciation override.
- `punctuation.py` shows PiperG2P processing prepared punctuation.
- `stream.py` writes request-local sentence-group chunks incrementally.
- `all_voices.py` lists catalog metadata and optionally synthesizes one voice.
