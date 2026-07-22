# 100-user trajectory extraction provenance

This note supplements, but does not rewrite, the immutable extraction manifest.

The full five-action extraction ran from `2026-07-13 11:21:48 EDT` to
`2026-07-13 12:08:38 EDT` (2,809.904 s) with:

```text
python -u preprocess/extract_hmog_trajectories.py \
  --output-dir results/trajectories_full_v2 \
  --max-users 100 \
  --confirm-full-run
```

The source archive resolves to
`/mnt/share/mwang49/Human_agent/hmog_dataset.zip`, is 6,132,356,276 bytes, and
has SHA-256
`4e3f4216ca7c362bd06493301d7ef9634940af69f939fe02689cb3f84c914346`.
The event preprocessor and configuration hashes already embedded in the
manifest were independently rechecked.

The canonical independent audit is
`results/trajectories_full_v2/formal_audit/formal_data_audit.json`; it reports
`formal_passed=true`. Its SHA-256 is
`ab2ca44f54cd221272faff46b092551486dda422dc251c814a48b62402c157b2`.

## Extractor-source limitation

The extractor launch hash was recorded before start as
`243767dc028049f01de8d312744c8fb01cc81330ba838660b5693f896e0fe391`.
The exact launch source bytes were not copied into the output directory. While
the process was running, the on-disk source received a documentation-only edit
and then hashed to
`c7a1df141d9e5a1d47ae1784e7de1490fcd6a894eff04167384b7f206783b9d3`;
the already-running Python process continued using the code loaded at launch.
This limitation is recorded explicitly rather than claiming a stronger source
binding than exists.

After extraction, version 1.1.1 fixed only the behavior of future
`--actions` subset runs: omitted higher-priority labels now still reserve their
raw contacts. The formal run requested all five actions and is therefore not
affected. The formal NPZ files and original manifest were not rewritten.

Machine-readable details and every output hash are in
`results/trajectories_full_v2/formal_audit/supplemental_provenance.json`.
