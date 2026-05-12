# GRU_0512

Semi-conductor APC/LiveData/log alignment utilities for GRU training.

## Current Alignment Entry Point

Build livedata-window aligned TSV files on a fixed `110ms` grid:

```bash
uv run python scripts/build_aligned_livedata_110ms.py --output outputs/aligned_livedata_110ms --overwrite
```

The generated aligned TSV files, raw dataset, caches, checkpoints, and logs are intentionally excluded from Git.

## Main Outputs

- `outputs/aligned_livedata_110ms/aligned/*.tsv`
- `outputs/aligned_livedata_110ms/manifest.json`
- `outputs/aligned_livedata_110ms/quality_report.md`

## Notes

- Final aligned rows are emitted only inside valid `livedata` coverage.
- Actions outside `livedata` coverage are still processed to preserve `state_*`, then dropped from final rows.
- Sensors use fixed `1..150` columns; `1..148` are real sampled sensors and `149..150` are default-filled sensors.
- `mask_1..mask_150` columns are retained in aligned TSV output.
