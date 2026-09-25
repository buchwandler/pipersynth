---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0007
release_version: v0.1.3
kind: changed
summary:
  Changed plan rendering to add prepare_plan_segments for canonical speech-only
  segments on the utterplan 0.2 line
status: rejected
audience: null
scopes: []
source_refs:
  - git:6e1d76d55301e228bb9951a32c56e5c02e989525
paths:
  - pipersynth/pipeline.py
  - pipersynth/__init__.py
  - pyproject.toml
issues: []
prs: []
sources:
  - git:6e1d76d55301e228bb9951a32c56e5c02e989525
contributors:
  - "@holgern"
breaking: false
internal: false
order: 7
---

PreparedAudioSegments renders immutable plan segments without semantic pause padding or prosody for canonical output. Dependency bounds now require utterplan 0.2 and piperg2p 0.1.6 or newer.
