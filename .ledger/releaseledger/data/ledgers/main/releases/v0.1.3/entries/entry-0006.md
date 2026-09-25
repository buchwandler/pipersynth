---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0006
release_version: v0.1.3
kind: docs
summary:
  Documented Piper catalog voice discovery and model-speaker identities in
  an all-voices example
status: accepted
audience: null
scopes: []
source_refs:
  - git:fd848fa659ea63768051c01313e3d82e79ec0d1b
paths:
  - examples/all_voices.py
  - examples/README.md
  - examples/run_all.py
issues: []
prs: []
sources:
  - git:fd848fa659ea63768051c01313e3d82e79ec0d1b
contributors:
  - "@holgern"
breaking: false
internal: false
order: 6
---

The all-voices example lists catalog metadata without downloading models, writes a local inventory, and optionally synthesizes one prepared sample for a selected voice and model speaker.
