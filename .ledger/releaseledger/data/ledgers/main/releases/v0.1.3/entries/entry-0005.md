---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0005
release_version: v0.1.3
kind: quality
summary:
  Improved voice-level benchmark reports with repeatable model-speaker measurements
  and phase-tagged failures
status: accepted
audience: null
scopes: []
source_refs:
  - git:b87a4e9bfde6d6c7c2d4f2d9c29a54cd9ff9e9a6
paths:
  - benchmarks/voice_level_benchmark.py
  - tests/test_voice_level_benchmark.py
issues: []
prs: []
sources:
  - git:b87a4e9bfde6d6c7c2d4f2d9c29a54cd9ff9e9a6
  - git:fd848fa659ea63768051c01313e3d82e79ec0d1b
contributors:
  - "@holgern"
breaking: false
internal: false
order: 5
---

The benchmark measures prepared stimuli for selected catalog voice and speaker identities, reuses a voice across identities for each model, and writes structured JSON reports. Voice-opening and synthesis failures retain explicit phase and error-type fields; no document mastering is performed.
