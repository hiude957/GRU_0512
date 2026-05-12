# Repo Data Processing

This note explains the current repo implementation from aligned TSV files to cached numpy arrays.

Important: merge rules have been updated to a fixed `110ms` grid limited to `livedata` coverage for final training/output rows. Action outside `livedata` coverage must still be processed in a hidden chronological state pass before those rows are filtered out. If current code or older notes still refer to inserting extra `log` rows or using `logapc` / `loglivedata` / `logonly` as action-bearing sources, treat that as legacy behavior that must be synchronized with `merge-rules.md`.

## 1. Scope

This reference covers the repo-local processing flow after raw `log` / `apc` / `livedata` have already been aligned into per-day TSV files.

Main entrypoints:

- `scripts/upgrade_semiconductor_alignment.py`
- `sensor_proxy.prepare_dataset`

It does not try to replace:

- `merge-rules.md` for the fixed merge logic
- `training-policy.md` for adjustable training-table policy

## 2. Artifact Ladder

Current practical ladder is:

1. raw anonymized files under `../datasets/anonymized/<date>/`
2. per-day aligned TSV files
3. upgraded aligned TSV files with `122` IO channels and repaired `state_*`
4. cached numpy arrays under `cache_sensor_proxy/<date>/`

The repo currently supports both:

- legacy per-date TSV layout like `./20260316/20260316_aligned_train.tsv`
- flat layout like `./trian/20260316_aligned_train.tsv`

## 3. What `upgrade_semiconductor_alignment.py` Does

This script upgrades already-aligned TSV files to the repo's current `122`-IO contract.

Key responsibilities:

- repair raw `2026-03-19` `livedata` sensor `99/121` complement rule before upgrading
- load IO default values from `../datasets/IO_Channel_index_default.xlsx`
- rebuild `evt_1..122` and `state_1..122`
- seed `state_*` from defaults, then carry state across rows and across days
- resume from a later `start-date` by seeding state from the previous upgraded day
- drop `livedata` rows whose timestamps fall inside APC intervals
- convert overlapped `loglivedata` rows to `logapc` if APC context exists
- drop stale `logapc` / `loglivedata` rows when the inherited sensor context is too old
- copy upgraded TSV files into `../datasets/trian/` when that flat directory exists

Current important constants:

- target IO count: `122`
- max action-row inherit gap: `1000 ms` by default

## 4. Upgraded Aligned TSV Contract

Important columns in the upgraded TSV:

- `ts_ms`
- `source_code`
- sensor columns `1..150`
- mask columns `mask_1..mask_150`
- `evt_1..122`
- `state_1..122`

Current sensor validity contract:

- keep exactly `150` sensor columns
- treat sensor `1..148` as normally collected real sensor dimensions
- treat sensor `149..150` as default-filled dimensions, not real target/loss dimensions
- before alignment, drop raw sensor rows that do not match the normal mask template `mask_1..mask_148 = 1` and `mask_149..mask_150 = 0`
- known abnormal input: `20260416` has `34` APC files with `58106` rows whose middle sensor masks are zero; those rows/files should be excluded from valid APC inputs instead of propagated into aligned TSV/cache

Legacy `source_code` meanings in older repo output:

- `1`: `apc`
- `2`: `livedata`
- `3`: `logapc`
- `4`: `loglivedata`

Fixed `110ms` grid output should instead map sensor provenance such as:

- `apc_grid`
- `livedata_grid`
- `carried_sensor`
- `sensor_default`
- `missing_sensor` only as an optional umbrella for invalid/no-real-sensor rows

Action presence should be represented by `evt_*`, `has_action`, or `action_count`, not by `source`.

For the current dataset start day `20260318`, final training/output rows should start from the first `livedata` coverage at `2026-03-18 11:14:23.321`. Rows from `00:00:00.000` until that first `livedata` are initialization-only: seed `state_*` from `action_default.xlsx`, process any hidden action updates if present, and do not emit pre-`livedata` training rows.

## 5. What `prepare_dataset` Reads

`sensor_proxy.prepare_dataset` loads one day's aligned TSV and supports either:

- `source_code` directly
- legacy string `source`

It also supports either:

- `ts_ms`
- or legacy `timestamp_raw`, which it parses into `ts_ms`

The legacy loader normalizes:

- source names to lowercase
- legacy source strings into numeric `source_code`
- timestamps into integer milliseconds

If `source` contains `logonly`, legacy rows are dropped before cache generation. Under the fixed `110ms` livedata-coverage rule, there should be no inserted `logonly` rows. Rows outside `livedata` coverage should normally be absent from the final training table after they have updated `state_*`; rows without trustworthy sensor support inside output coverage should use `carried_sensor` or `sensor_default` with `mask_* = 0`.

## 6. How Cache Arrays Are Built

For each day, `prepare_dataset` extracts:

- `sensors`
- `masks`
- `evt`
- `state`
- `source_code`
- `ts_ms`

Then it builds helper signals:

- `has_action`: whether any `evt_* != 0`
- `real_sensor_row`: whether `source_code in {1, 2}`
- `delta_t`: elapsed time from previous row, clipped by `time_feature_cap_ms`
- `since_action`
- `since_sensor`

The final model input feature matrix is:

`[sensors, masks, evt, state, source_onehot, time_features]`

Where:

- legacy `source_onehot` has 4 dims for source codes `1..4`
- `time_features` has 3 dims: `delta_t`, `since_action`, `since_sensor`

For fixed `110ms` grid output inside a continuous `livedata` coverage segment, `delta_t` should normally be constant. Across filtered hidden spans, `delta_t`, `since_action`, and `since_sensor` should preserve the real elapsed-time meaning rather than pretending the rows were adjacent. `source_onehot` must be regenerated from the new `source_code` mapping.

If abnormal raw sensor rows have already been dropped before alignment, `masks` can be generated from the fixed valid-sensor template for compatibility: `1` for sensor `1..148`, `0` for sensor `149..150`, with all-zero masks for any retained `carried_sensor` / `sensor_default` context rows. Whether `masks` are included as model input features can be an ablation choice, but `target_mask` / `loss_mask` semantics must still exclude `149..150`.

## 7. Future Target Semantics

For current row `row_i`, the target is:

- the next row after `row_i`
- whose `source_code` is `1` or `2`
- whose future gap is within `max_target_dt_ms`

From that future real-sensor row, the repo builds:

- `target_source`: future real source code (`1` or `2`)
- `target_sensor`: future full normalized sensor value, used by the current full-prediction GRU
- `target_delta`: `future_sensor - current_sensor`
- `target_log_dt`: `log1p(future_ts - current_ts)`
- `target_mask`: only dims where the future real-sensor mask is valid
- `loss_mask`: compatibility array for dims where both current and future masks are valid
- `valid_target`: future row exists, gap is valid, and at least one `target_mask` dim is valid

Rows that contain action inside the retained `livedata` coverage grid stay in inputs, but action presence is not a sensor target criterion. Hidden action rows outside `livedata` coverage are not saved as training rows after their `state_*` updates have been applied.

Under the fixed livedata-coverage grid policy, `carried_sensor` and `sensor_default` rows may stay in the input sequence only when they occur inside final output coverage as local input context; they are never real-sensor target rows. Their `mask_* = 0` also means they should not create valid loss dimensions. The `20260318` pre-`livedata` initialization window should not be emitted as training rows.

## 8. Saved Cache Files

Per-day arrays written under `cache_sensor_proxy/<date>/`:

- `features.npy`
- `target_sensor.npy`
- `target_mask.npy`
- `target_delta.npy`
- `target_log_dt.npy`
- `loss_mask.npy`
- `sample_weight.npy`
- `valid_target.npy`
- `ts_ms.npy`
- `source_code.npy`
- `target_source.npy`
- `has_action.npy`

Per-day metadata:

- `<cache_dir>/<date>.meta.json`

Whole-run manifest:

- `<cache_dir>/manifest.json`

## 9. Sample Weight Policy in Code

Current weight logic in `prepare_dataset`:

- start from target-source weights
- multiply by input-event weights

Target-source weights distinguish:

- `apc`
- `livedata`

Input-event weights distinguish:

- normal rows
- rows within `1s` after an action
- rows that contain an action

Rows with `valid_target = false` always get zero sample weight.

## 10. Recommended Explanation Boundary

When asked about the repo's current data processing flow:

1. explain raw sources and fixed merge rules using `merge-rules.md`
2. state that the target aligned TSV is now fixed to `livedata` coverage `110ms` grid rows, with hidden outside-coverage action updates applied before filtering
3. explain upgraded TSV contract and state inheritance
4. explain `prepare_dataset` cache construction, noting any legacy assumptions that must be updated
5. explain future-target semantics
6. separate fixed merge logic from adjustable training policy
