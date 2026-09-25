---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.0
kind: internal
summary: Documented the superseded dependency-floor adjustment for release audit
status: accepted
audience: null
scopes: []
source_refs:
  - git:09242788b1ceb8a8b299f1f4f8e9e52233e120ae
paths:
  - pyproject.toml
issues: []
prs: []
sources:
  - git:09242788b1ceb8a8b299f1f4f8e9e52233e120ae
contributors:
  - "@holgern"
breaking: false
internal: true
order: 2
---

The temporary PiperG2P minimum-version change was reversed before the v0.2.0 candidate. It has no net user-visible effect and is retained only to account for the audited commit.
