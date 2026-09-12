# Performance guidance

Measure these phases separately: cold import, voice configuration, ONNX session creation, frontend initialization, phonemization, inference, postprocessing, and total real-time factor.

Reuse one `PiperVoice` or `PiperPipeline` for repeated requests. Session creation is substantially more expensive than a synthesis call. Use `iter_units` or `prepare_units(...).render()` for long documents to keep rendered unit audio bounded. PiperSynth does not cache waveforms by default because synthesis settings and noise can change output.
