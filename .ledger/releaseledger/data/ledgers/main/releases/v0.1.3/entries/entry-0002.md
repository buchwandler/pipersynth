---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.3
kind: changed
summary:
  Changed complete-output rendering to apply target LUFS and true-peak ceiling
  policy for whole-document output
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - pipersynth/audio_job.py
  - pipersynth/pipeline.py
issues: []
prs: []
sources:
  - git:e4f1c6860f03ffec363d09ac2d03c0eeaad4aad4
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---

The AudioJob output policy forwards target_lufs, true_peak_ceiling_dbtp, and peak_policy to composition for whole-document output. Streaming unit preparation rejects target_lufs instead of normalizing units independently. Calibration mode, key, gain, and source are recorded in inference metadata and AudioResult metadata.
