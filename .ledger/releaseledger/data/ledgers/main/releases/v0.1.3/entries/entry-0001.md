---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0001
release_version: v0.1.3
kind: added
summary:
  Added static voice-level calibration with validated per-model and speaker
  gain data
status: accepted
audience: null
scopes: []
source_refs:
  - git:e4f1c6860f03ffec363d09ac2d03c0eeaad4aad4
paths:
  - pipersynth/voice_level.py
  - pipersynth/voice.py
  - pipersynth/__init__.py
  - pipersynth/data/voice_level_calibration.json
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

VoiceLevelConfig supports deterministic engine-local gain adjustments keyed by exact Piper model, quality, and speaker identities. The packaged catalog is validated offline. Calibration applies after optional request peak normalization and before explicit output_gain; PiperSynth does not measure LUFS at runtime or master complete documents.
