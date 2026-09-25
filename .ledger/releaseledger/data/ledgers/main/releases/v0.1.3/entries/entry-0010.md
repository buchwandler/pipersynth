---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 5
entry_id: entry-0010
release_version: v0.1.3
kind: changed
summary:
  Changed PiperSynth to a standalone engine with protected regex chunking for
  prepared text
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0014
  - tl:task-0015
  - git:6e1d76d55301e228bb9951a32c56e5c02e989525
paths:
  - README.md
  - docs/architecture.md
  - docs/performance.md
  - examples/README.md
  - examples/long_text.py
  - pyproject.toml
  - pipersynth/__init__.py
  - pipersynth/convenience.py
  - pipersynth/types.py
  - pipersynth/voice.py
  - pipersynth/pipeline.py
  - pipersynth/planning.py
  - pipersynth/preparation.py
  - pipersynth/composition.py
  - tests/test_engine_public_api.py
  - tests/test_engine_voice.py
issues: []
prs: []
sources: []
contributors: []
breaking: true
internal: false
order: 10
---

Callers provide prepared, speakable text. TextChunkingConfig defaults to Phrasplit regex sentence splitting and supports max_chars; mode=none bypasses only PiperSynth pre-segmentation while PiperG2P retains its normal sentence groups. Split boundaries are merged across pronunciation overrides, token annotations, and raw phoneme blocks, preserving exact source offsets. PiperSynth joins inferred audio in source order without semantic silence, document timelines, or final mastering. This remains a breaking API and dependency-boundary change.
