---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: v0.1.3
kind: added
summary:
  Added the packaged voice loudness calibration catalog for measured Piper
  speaker identities at -24 LUFS
status: accepted
audience: null
scopes: []
source_refs:
  - git:0208866f807084e80142feb3bd6caba9f2eb1af3
paths:
  - pipersynth/data/voice_level_calibration.json
  - docs/performance.md
issues: []
prs: []
sources:
  - git:0208866f807084e80142feb3bd6caba9f2eb1af3
  - git:e4f1c6860f03ffec363d09ac2d03c0eeaad4aad4
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---

The packaged catalog records BS.1770 gain, measured LUFS, and variability for identities measured from corpus pipersynth-count-1-to-10-v1 with a -1 dBTP calibration ceiling. docs/performance.md documents that Vietnamese and Chinese identities must be remeasured after PiperG2P phoneme coverage changes.
