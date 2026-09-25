---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 5
entry_id: entry-0001
release_version: v0.2.0
kind: changed
summary:
  Changed PiperSynth from PiperPipeline to PiperVoice.synthesize(SynthesisRequest);
  callers split text and compose output
status: accepted
audience: null
scopes: []
source_refs:
  - git:2731e013a09cdc785c045e4f70c03d511f50428f
  - git:7a6acb03a1a505cda494966e3a98e3448bb7695e
  - git:a13812149614374affaf1867ef221b8b0c2704aa
paths:
  - README.md
  - docs/architecture.md
  - docs/troubleshooting.md
  - pyproject.toml
  - pipersynth/__init__.py
  - pipersynth/errors.py
  - pipersynth/types.py
  - pipersynth/voice.py
  - pipersynth/voice_level.py
  - tests/test_engine_public_api.py
  - tests/test_engine_voice.py
issues: []
prs: []
sources:
  - git:2731e013a09cdc785c045e4f70c03d511f50428f
  - git:7a6acb03a1a505cda494966e3a98e3448bb7695e
  - git:a13812149614374affaf1867ef221b8b0c2704aa
contributors:
  - "@holgern"
breaking: true
internal: false
order: 1
---

PiperSynth 0.2.0 accepts prepared, speakable text as atomic synthesis requests, not documents or plans. The former public APIs PiperPipeline, PipelineConfig, GenerationConfig, UtterancePlan, UtterancePlanner, AudioResult, AudioUnitResult, LoudnessConfig, and PreparedTextResult are removed; document planning, text preparation, and composition belong elsewhere. The UtterPlan, AudioCompose, and SSMD runtime dependencies are also removed. To migrate from 0.1.x, move document parsing, written-form expansion, and planning to the caller; create one SynthesisRequest per desired speech unit and call PiperVoice.synthesize(), or use synthesize_text() for one prepared request. Each call performs one acoustic inference and does not choose text boundaries or retry an over-capacity request. Split long text in the caller and handle SynthesisInputTooLongError. Compose semantic pauses and timelines, and perform final mastering, in a downstream layer such as AudioCompose. Static per-identity voice-level calibration remains opt-in and is not final LUFS or true-peak mastering.
