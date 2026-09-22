---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0005
release_version: v0.1.3
kind: quality
summary:
  Improved loudness benchmark measurement with isolated per-model workers and
  phase-tagged failure reporting
status: accepted
audience: null
scopes: []
source_refs:
  - git:b87a4e9bfde6d6c7c2d4f2d9c29a54cd9ff9e9a6
paths:
  - benchmarks/voice_loudness.py
  - tests/test_voice_loudness_benchmark.py
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

One pipeline is opened per model and reused across its speaker identities. Each model runs in a fresh worker process with atomic JSON results. Failures carry phase and error_type fields, non-finite metrics are rejected, and locale count stimuli are resolved in a preflight before any synthesis.
