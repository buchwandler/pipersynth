---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0006
release_version: v0.1.3
kind: docs
summary:
  Documented catalog voices, per-speaker calibration identities, and Spokenform
  language gaps in an all-voices example
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

examples/all_voices.py prints the catalog without downloading voice models and writes inventory, language, and missing-language reports below example-artefacts. The full showcase is resource-heavy and runs only with an explicit runner flag.
