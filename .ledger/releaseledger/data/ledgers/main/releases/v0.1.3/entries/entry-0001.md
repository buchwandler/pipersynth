---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.3
kind: added
summary:
  Added calibrated voice leveling with LoudnessConfig and a validated offline
  per-voice gain catalog
status: accepted
audience: null
scopes: []
source_refs:
  - git:e4f1c6860f03ffec363d09ac2d03c0eeaad4aad4
paths:
  - pipersynth/loudness_config.py
  - pipersynth/voice_level.py
  - pipersynth/voice.py
  - pipersynth/__init__.py
issues: []
prs: []
sources:
  - git:e4f1c6860f03ffec363d09ac2d03c0eeaad4aad4
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

LoudnessConfig configures voice leveling, target LUFS, true-peak ceiling, peak policy, and an explicit voice gain on pipeline and synthesis configuration. Calibration lookup uses exact piper model, quality, and speaker identities from a strictly validated offline catalog and applies one deterministic static gain after legacy peak normalization. Synthesis never measures LUFS at runtime and missing identities are zero-gain diagnostic no-ops.
