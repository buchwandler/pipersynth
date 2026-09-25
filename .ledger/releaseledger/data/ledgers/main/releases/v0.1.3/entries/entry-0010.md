---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0010
release_version: v0.1.3
kind: changed
summary:
  Changed PiperSynth from a document-oriented pipeline into a standalone PiperVoice
  synthesis engine
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0014
paths:
  - README.md
  - pyproject.toml
  - pipersynth/voice.py
  - pipersynth/types.py
  - pipersynth/__init__.py
  - pipersynth/pipeline.py
  - pipersynth/planning.py
  - pipersynth/preparation.py
  - pipersynth/composition.py
issues: []
prs: []
sources: []
contributors: []
breaking: true
internal: false
order: 10
---

The public API accepts prepared request text, returns independent rendered audio, and streams request-local sentence chunks. Removed document planning, semantic text preparation, composition, and their legacy APIs from PiperSynth; callers retain control over document timelines and final mastering. This is a breaking API and dependency-boundary change.
