#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SENSOR_COUNT = 150
IO_COUNT = 122
SOURCE_ONEHOT_COUNT = 4
TIME_FEATURE_COUNT = 3
FEATURE_DIM_NO_MASK = SENSOR_COUNT + IO_COUNT + IO_COUNT + SOURCE_ONEHOT_COUNT + TIME_FEATURE_COUNT

SENSOR_START = 0
MASK_START = SENSOR_START + SENSOR_COUNT
EVT_START = MASK_START + SENSOR_COUNT
STATE_START = EVT_START + IO_COUNT
HAS_ACTION_INDEX = STATE_START + IO_COUNT
ACTION_COUNT_INDEX = HAS_ACTION_INDEX + 1
NUMERIC_COLUMN_COUNT = ACTION_COUNT_INDEX + 1

FEATURE_SENSOR_START = 0
FEATURE_EVT_START = FEATURE_SENSOR_START + SENSOR_COUNT
FEATURE_STATE_START = FEATURE_EVT_START + IO_COUNT
FEATURE_SOURCE_START = FEATURE_STATE_START + IO_COUNT
FEATURE_TIME_START = FEATURE_SOURCE_START + SOURCE_ONEHOT_COUNT

REAL_SENSOR_SOURCES = {1, 2}


@dataclass
class DayRange:
    date: str
    start: int
    end: int
    rows: int
    first_ts_ms: int | None
    last_ts_ms: int | None


@dataclass
class SplitStats:
    split: str
    rows: int
    valid_targets: int
    first_ts_ms: int | None
    last_ts_ms: int | None
    dates: list[str]
    feature_dim: int
    feature_columns: list[str]
    target_sensor_dim: int
    input_mask_saved: bool
    masks_in_features: bool
    max_target_dt_ms: int
    time_feature_cap_ms: int
    day_ranges: list[DayRange]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GRU cache arrays from livedata-window aligned TSV files."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=Path("outputs/aligned_livedata_110ms/aligned"),
        help="Directory containing YYYYMMDD_aligned.tsv files.",
    )
    parser.add_argument(
        "--aligned-manifest",
        type=Path,
        default=Path("outputs/aligned_livedata_110ms/manifest.json"),
        help="Aligned manifest with per-day output row counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cache/gru_110ms_nomask_val_20260508_20260511"),
        help="Output cache directory.",
    )
    parser.add_argument("--val-start", default="20260508")
    parser.add_argument("--val-end", default="20260511")
    parser.add_argument(
        "--max-target-dt-ms",
        type=int,
        default=2000,
        help="A next-row target is valid only when its timestamp gap is <= this value.",
    )
    parser.add_argument(
        "--time-feature-cap-ms",
        type=int,
        default=3_600_000,
        help="Clip delta/since time features to this value, then divide by it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return sorted(manifest["days"], key=lambda item: item["date"])


def split_days(days: list[dict], val_start: str, val_end: str) -> dict[str, list[dict]]:
    train = [day for day in days if day["date"] < val_start]
    val = [day for day in days if val_start <= day["date"] <= val_end]
    return {"train": train, "val": val}


def feature_columns() -> list[str]:
    return (
        [f"sensor_{i}" for i in range(1, SENSOR_COUNT + 1)]
        + [f"evt_{i}" for i in range(1, IO_COUNT + 1)]
        + [f"state_{i}" for i in range(1, IO_COUNT + 1)]
        + [f"source_code_{i}" for i in range(1, SOURCE_ONEHOT_COUNT + 1)]
        + ["delta_t", "since_action", "since_sensor"]
    )


def allocate_arrays(split_dir: Path, rows: int) -> dict[str, np.memmap]:
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "features": np.lib.format.open_memmap(
            split_dir / "features.npy", mode="w+", dtype=np.float32, shape=(rows, FEATURE_DIM_NO_MASK)
        ),
        "input_mask": np.lib.format.open_memmap(
            split_dir / "input_mask.npy", mode="w+", dtype=np.uint8, shape=(rows, SENSOR_COUNT)
        ),
        "target_sensor": np.lib.format.open_memmap(
            split_dir / "target_sensor.npy", mode="w+", dtype=np.float32, shape=(rows, SENSOR_COUNT)
        ),
        "target_delta": np.lib.format.open_memmap(
            split_dir / "target_delta.npy", mode="w+", dtype=np.float32, shape=(rows, SENSOR_COUNT)
        ),
        "target_log_dt": np.lib.format.open_memmap(
            split_dir / "target_log_dt.npy", mode="w+", dtype=np.float32, shape=(rows,)
        ),
        "target_mask": np.lib.format.open_memmap(
            split_dir / "target_mask.npy", mode="w+", dtype=np.uint8, shape=(rows, SENSOR_COUNT)
        ),
        "loss_mask": np.lib.format.open_memmap(
            split_dir / "loss_mask.npy", mode="w+", dtype=np.uint8, shape=(rows, SENSOR_COUNT)
        ),
        "sample_weight": np.lib.format.open_memmap(
            split_dir / "sample_weight.npy", mode="w+", dtype=np.float32, shape=(rows,)
        ),
        "valid_target": np.lib.format.open_memmap(
            split_dir / "valid_target.npy", mode="w+", dtype=np.bool_, shape=(rows,)
        ),
        "ts_ms": np.lib.format.open_memmap(
            split_dir / "ts_ms.npy", mode="w+", dtype=np.int64, shape=(rows,)
        ),
        "source_code": np.lib.format.open_memmap(
            split_dir / "source_code.npy", mode="w+", dtype=np.int16, shape=(rows,)
        ),
        "target_source": np.lib.format.open_memmap(
            split_dir / "target_source.npy", mode="w+", dtype=np.int16, shape=(rows,)
        ),
        "has_action": np.lib.format.open_memmap(
            split_dir / "has_action.npy", mode="w+", dtype=np.bool_, shape=(rows,)
        ),
        "action_count": np.lib.format.open_memmap(
            split_dir / "action_count.npy", mode="w+", dtype=np.int16, shape=(rows,)
        ),
        "date_index": np.lib.format.open_memmap(
            split_dir / "date_index.npy", mode="w+", dtype=np.int16, shape=(rows,)
        ),
    }


def parse_aligned_numeric(line: str) -> tuple[int, int, np.ndarray]:
    prefix = line.rstrip("\n").split("\t", 4)
    if len(prefix) != 5:
        raise ValueError("Aligned TSV row has fewer than 5 columns")
    ts_ms = int(prefix[1])
    source_code = int(prefix[3])
    numeric = np.fromstring(prefix[4], sep="\t", dtype=np.float32)
    if numeric.shape[0] != NUMERIC_COLUMN_COUNT:
        raise ValueError(f"Expected {NUMERIC_COLUMN_COUNT} numeric columns, got {numeric.shape[0]}")
    return ts_ms, source_code, numeric


def normalized_time(value_ms: int | float, cap_ms: int) -> float:
    if not math.isfinite(float(value_ms)):
        return 1.0
    return min(max(float(value_ms), 0.0), float(cap_ms)) / float(cap_ms)


def fill_previous_target(
    arrays: dict[str, np.memmap],
    prev_idx: int | None,
    current_idx: int,
    current_ts_ms: int,
    current_source_code: int,
    current_sensor: np.ndarray,
    current_mask: np.ndarray,
    max_target_dt_ms: int,
) -> bool:
    if prev_idx is None:
        return False
    prev_ts_ms = int(arrays["ts_ms"][prev_idx])
    gap = current_ts_ms - prev_ts_ms
    if gap <= 0 or gap > max_target_dt_ms:
        return False
    if current_source_code not in REAL_SENSOR_SOURCES:
        return False
    target_mask = current_mask.astype(np.uint8, copy=False)
    if int(target_mask.sum()) <= 0:
        return False
    arrays["target_sensor"][prev_idx] = current_sensor
    arrays["target_delta"][prev_idx] = current_sensor - arrays["features"][prev_idx, FEATURE_SENSOR_START:FEATURE_EVT_START]
    arrays["target_log_dt"][prev_idx] = math.log1p(gap)
    arrays["target_mask"][prev_idx] = target_mask
    arrays["loss_mask"][prev_idx] = np.logical_and(arrays["input_mask"][prev_idx], target_mask).astype(np.uint8)
    arrays["sample_weight"][prev_idx] = 1.0
    arrays["valid_target"][prev_idx] = True
    arrays["target_source"][prev_idx] = current_source_code
    return True


def iter_nonempty_days(days: Iterable[dict]) -> Iterable[dict]:
    for day in days:
        if int(day["output_rows"]) > 0:
            yield day


def build_split(
    split: str,
    days: list[dict],
    aligned_dir: Path,
    out_dir: Path,
    max_target_dt_ms: int,
    time_feature_cap_ms: int,
) -> SplitStats:
    rows = sum(int(day["output_rows"]) for day in days)
    split_dir = out_dir / split
    arrays = allocate_arrays(split_dir, rows)

    columns = feature_columns()
    row_idx = 0
    prev_idx: int | None = None
    prev_ts_ms: int | None = None
    last_action_ts_ms: int | None = None
    last_sensor_ts_ms: int | None = None
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None
    valid_targets = 0
    day_ranges: list[DayRange] = []
    date_labels: list[str] = []

    for date_idx, day in enumerate(days):
        date = day["date"]
        date_labels.append(date)
        expected_rows = int(day["output_rows"])
        day_start_idx = row_idx
        day_first_ts: int | None = None
        day_last_ts: int | None = None
        path = aligned_dir / f"{date}_aligned.tsv"
        if expected_rows == 0:
            day_ranges.append(DayRange(date, row_idx, row_idx, 0, None, None))
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            if len(header) < 5 or header[1] != "ts_ms" or header[3] != "source_code":
                raise ValueError(f"Unexpected aligned header in {path}")
            for line in fh:
                ts_ms, source_code, numeric = parse_aligned_numeric(line)
                sensor = numeric[SENSOR_START:MASK_START]
                mask = numeric[MASK_START:EVT_START].astype(np.uint8)
                evt = numeric[EVT_START:STATE_START]
                state = numeric[STATE_START:HAS_ACTION_INDEX]
                has_action = bool(int(numeric[HAS_ACTION_INDEX]))
                action_count = int(numeric[ACTION_COUNT_INDEX])

                if row_idx >= rows:
                    raise ValueError(f"{split} has more rows than manifest declares")
                if first_ts_ms is None:
                    first_ts_ms = ts_ms
                if day_first_ts is None:
                    day_first_ts = ts_ms

                delta_t_ms = 0 if prev_ts_ms is None else ts_ms - prev_ts_ms
                since_action_ms = 0 if has_action else (
                    time_feature_cap_ms if last_action_ts_ms is None else ts_ms - last_action_ts_ms
                )
                real_sensor_row = source_code in REAL_SENSOR_SOURCES and int(mask.sum()) > 0
                since_sensor_ms = 0 if real_sensor_row else (
                    time_feature_cap_ms if last_sensor_ts_ms is None else ts_ms - last_sensor_ts_ms
                )

                arrays["features"][row_idx, FEATURE_SENSOR_START:FEATURE_EVT_START] = sensor
                arrays["features"][row_idx, FEATURE_EVT_START:FEATURE_STATE_START] = evt
                arrays["features"][row_idx, FEATURE_STATE_START:FEATURE_SOURCE_START] = state
                arrays["features"][row_idx, FEATURE_SOURCE_START:FEATURE_TIME_START] = 0.0
                if 1 <= source_code <= SOURCE_ONEHOT_COUNT:
                    arrays["features"][row_idx, FEATURE_SOURCE_START + source_code - 1] = 1.0
                arrays["features"][row_idx, FEATURE_TIME_START:FEATURE_TIME_START + TIME_FEATURE_COUNT] = (
                    normalized_time(delta_t_ms, time_feature_cap_ms),
                    normalized_time(since_action_ms, time_feature_cap_ms),
                    normalized_time(since_sensor_ms, time_feature_cap_ms),
                )
                arrays["input_mask"][row_idx] = mask
                arrays["ts_ms"][row_idx] = ts_ms
                arrays["source_code"][row_idx] = source_code
                arrays["has_action"][row_idx] = has_action
                arrays["action_count"][row_idx] = action_count
                arrays["date_index"][row_idx] = date_idx

                if fill_previous_target(
                    arrays,
                    prev_idx,
                    row_idx,
                    ts_ms,
                    source_code,
                    sensor,
                    mask,
                    max_target_dt_ms,
                ):
                    valid_targets += 1

                prev_idx = row_idx
                prev_ts_ms = ts_ms
                if has_action:
                    last_action_ts_ms = ts_ms
                if real_sensor_row:
                    last_sensor_ts_ms = ts_ms
                day_last_ts = ts_ms
                last_ts_ms = ts_ms
                row_idx += 1

        actual_day_rows = row_idx - day_start_idx
        if actual_day_rows != expected_rows:
            raise ValueError(f"{date} manifest rows={expected_rows}, read rows={actual_day_rows}")
        day_ranges.append(DayRange(date, day_start_idx, row_idx, actual_day_rows, day_first_ts, day_last_ts))
        print(
            f"{split}: {date} rows={actual_day_rows} cumulative={row_idx}/{rows}",
            flush=True,
        )

    if row_idx != rows:
        raise ValueError(f"{split} manifest rows={rows}, read rows={row_idx}")
    for array in arrays.values():
        array.flush()

    with (split_dir / "dates.json").open("w", encoding="utf-8") as fh:
        json.dump(date_labels, fh, ensure_ascii=False, indent=2)
    stats = SplitStats(
        split=split,
        rows=rows,
        valid_targets=valid_targets,
        first_ts_ms=first_ts_ms,
        last_ts_ms=last_ts_ms,
        dates=date_labels,
        feature_dim=FEATURE_DIM_NO_MASK,
        feature_columns=columns,
        target_sensor_dim=SENSOR_COUNT,
        input_mask_saved=True,
        masks_in_features=False,
        max_target_dt_ms=max_target_dt_ms,
        time_feature_cap_ms=time_feature_cap_ms,
        day_ranges=day_ranges,
    )
    with (split_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(stats), fh, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; use --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    days = load_manifest(args.aligned_manifest)
    splits = split_days(days, args.val_start, args.val_end)
    stats: dict[str, SplitStats] = {}
    for split in ("train", "val"):
        print(f"Building {split} cache", flush=True)
        stats[split] = build_split(
            split=split,
            days=splits[split],
            aligned_dir=args.aligned_dir,
            out_dir=args.output,
            max_target_dt_ms=args.max_target_dt_ms,
            time_feature_cap_ms=args.time_feature_cap_ms,
        )

    manifest = {
        "aligned_dir": str(args.aligned_dir),
        "aligned_manifest": str(args.aligned_manifest),
        "output": str(args.output),
        "val_start": args.val_start,
        "val_end": args.val_end,
        "feature_dim": FEATURE_DIM_NO_MASK,
        "masks_in_features": False,
        "input_mask_saved": True,
        "feature_layout": {
            "sensors": [FEATURE_SENSOR_START, FEATURE_EVT_START],
            "evt": [FEATURE_EVT_START, FEATURE_STATE_START],
            "state": [FEATURE_STATE_START, FEATURE_SOURCE_START],
            "source_onehot": [FEATURE_SOURCE_START, FEATURE_TIME_START],
            "time_features": [FEATURE_TIME_START, FEATURE_TIME_START + TIME_FEATURE_COUNT],
        },
        "source_code": {
            "1": "apc_grid",
            "2": "livedata_grid",
            "3": "carried_sensor",
            "4": "sensor_default",
        },
        "real_sensor_source_codes": sorted(REAL_SENSOR_SOURCES),
        "target_policy": {
            "target": "next real-sensor row inside the same split",
            "max_target_dt_ms": args.max_target_dt_ms,
            "sensor_dims": SENSOR_COUNT,
            "loss_dims": "mask-valid dims only; sensors 149-150 are normally masked out",
        },
        "splits": {name: asdict(value) for name, value in stats.items()},
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"Wrote cache manifest to {args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
