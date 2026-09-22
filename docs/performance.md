# Performance guidance

Measure these phases separately: cold import, voice configuration, OnnxVoice installation, runtime opening, frontend initialization, phonemization, inference, composition, and total real-time factor.

Reuse one `PiperVoice` or `PiperPipeline` for repeated requests. OnnxVoice runtime opening is substantially more expensive than a synthesis call. `render_plan()` is batch-oriented: it renders the plan once, builds one AudioJob, and composes it once. `to_audio_job()` is useful when persistence or generic replay is needed, but it does not avoid inference.

Use `iter_units()` or `prepare_units(...).render()` for streaming-style long documents to keep rendered unit audio bounded. Streaming remains separate from complete-output composition and whole-document loudness policy. PiperSynth does not cache waveforms by default because synthesis settings and model noise can change output.

## Voice loudness calibration maintenance

The current loudness calibration promotes measured Vietnamese and Chinese identities even when the benchmark reports non-fatal `MissingPhonemeWarning` messages. After PiperG2P phoneme coverage changes for Vietnamese or Chinese text voices, rerun the affected loudness identities because corrected phonemization may alter the synthesized signal and its measured loudness.
