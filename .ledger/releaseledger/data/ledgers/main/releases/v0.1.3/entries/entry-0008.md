---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0008
release_version: v0.1.3
kind: changed
summary:
  Improved prepared units with linguistic run provenance and morphological
  token annotations clipped to segment spans
status: rejected
audience: null
scopes: []
source_refs: []
paths:
  - pipersynth/plan_adapter.py
issues: []
prs: []
sources:
  - git:6e1d76d55301e228bb9951a32c56e5c02e989525
contributors:
  - "@holgern"
breaking: false
internal: false
order: 8
---

Token annotations are clipped to the segment span instead of dropped when they cross boundaries. Frontends that cannot accept token annotations now raise PlanRenderingError instead of silently ignoring them.
