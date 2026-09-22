---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: v0.1.3
kind: quality
summary:
  Improved calibration promotion with coverage count validation and reviewed
  partial promotion
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - benchmarks/voice_loudness_calibration.py
issues: []
prs: []
sources:
  - git:0208866f807084e80142feb3bd6caba9f2eb1af3
  - git:b87a4e9bfde6d6c7c2d4f2d9c29a54cd9ff9e9a6
contributors:
  - "@holgern"
breaking: false
internal: false
order: 4
---

build_runtime_calibration rejects inconsistent coverage counts, promotes explicitly reviewed statuses without the MAD gate, and requires an explicit opt-in for partial coverage instead of silently accepting incomplete reports.
