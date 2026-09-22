---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0009
release_version: v0.1.3
kind: internal
summary:
  Improved formatting and lint consistency across benchmarks, examples, and
  tests
status: accepted
audience: null
scopes: []
source_refs:
  - git:99bb8364904dc20e34f2c90f53d09c54c1d99312
paths:
  - benchmarks/voice_loudness.py
  - examples/README.md
  - examples/all_voices.py
  - examples/run_all.py
  - tests/test_all_voices_example.py
  - tests/test_examples_runner.py
  - tests/test_voice_loudness_benchmark.py
issues: []
prs: []
sources:
  - git:99bb8364904dc20e34f2c90f53d09c54c1d99312
contributors:
  - "@holgern"
breaking: false
internal: true
order: 9
---

Housekeeping only: pre-commit formatting and example test adjustments with no user-visible behavior change. Retained as an internal entry for commit coverage and excluded from the public changelog.
