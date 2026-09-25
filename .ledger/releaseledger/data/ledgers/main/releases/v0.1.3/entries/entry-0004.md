---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0004
release_version: v0.1.3
kind: quality
summary:
  Improved calibration promotion with coverage validation and explicit partial-coverage
  review
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - benchmarks/promote_voice_calibration.py
  - tests/test_voice_level_benchmark.py
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

Candidate catalog generation checks that reported identity counts, completeness flags, repeat counts, policy-derived gains, and eligibility statuses agree. Partial coverage requires an explicit opt-in, and promotion writes a separate candidate rather than the packaged production catalog.
