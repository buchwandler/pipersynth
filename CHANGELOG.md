# Changelog

## Unreleased

- Begin the production PiperSynth implementation with bounded piperg2p compatibility, typed packaging, and independent ONNX runtime support.
- Integrate TTSPlan as the semantic planning layer for PiperPipeline, including reusable plans, PlanUnit streaming, resolved pauses, SSMD, and Spokenform.
- Add direct phoneme mode, plan provenance in results and diagnostics, explicit directive and language errors, and planner-aware convenience controls.
- Deprecate PiperSynth-owned semantic sentence silence and the direct Spokenform extra.
