---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.3
kind: added
summary: Added a versioned, model-free request API contract for integrations
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0020
paths:
  - pipersynth/api_contract.py
  - pipersynth/__init__.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

The public contract identifies PiperVoice.synthesize as the canonical atomic request entrypoint and records supported request capabilities without requiring a loaded voice model.
