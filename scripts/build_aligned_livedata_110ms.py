#!/usr/bin/env python3
"""Build livedata-window 110ms aligned TSV files.

The alignment policy is documented in:
docs/action_outside_livedata_coverage.md
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


GRID_STEP_MS = 110
APC_GAP_MS = 250
APC_NEAREST_MS = 125
LIVEDATA_GAP_MS = 2000
SENSOR_COUNT = 150
REAL_SENSOR_COUNT = 148
IO_COUNT = 122
EPOCH = datetime(1970, 1, 1)
NORMAL_MASK = ["1"] * REAL_SENSOR_COUNT + ["0"] * (SENSOR_COUNT - REAL_SENSOR_COUNT)
ALL_ZERO_MASK = ["0"] * SENSOR_COUNT
SOURCE_CODE = {
    "apc_grid": 1,
    "livedata_grid": 2,
    "carried_sensor": 3,
    "sensor_default": 4,
}


@dataclass(frozen=True)
class SensorRow:
    ts_ms: int
    seq: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class SensorSeries:
    times: list[int]
    values: list[tuple[float, ...]]
    seg_ids: list[int]
    gap_ms: int
    source_name: str


@dataclass
class DayStats:
    date: str
    livedata_segments: int = 0
    output_rows: int = 0
    hidden_actions: int = 0
    retained_actions: int = 0
    rejected_apc_rows: int = 0
    accepted_apc_rows: int = 0
    rejected_livedata_rows: int = 0
    accepted_livedata_rows: int = 0
    source_counts: Counter = None

    def __post_init__(self) -> None:
        if self.source_counts is None:
            self.source_counts = Counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", type=Path)
    parser.add_argument("--output", default="outputs/aligned_livedata_110ms", type=Path)
    parser.add_argument("--dates", default="", help="Comma-separated YYYYMMDD dates or ranges like 20260416-20260418")
    parser.add_argument("--start-date", default="", help="YYYYMMDD inclusive")
    parser.add_argument("--end-date", default="", help="YYYYMMDD inclusive")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dt_to_ms(dt: datetime) -> int:
    delta = dt - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000) + (delta.microseconds // 1000)


def ms_to_dt(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=ms)


def day_start_ms(day: str) -> int:
    return dt_to_ms(datetime.strptime(day, "%Y%m%d"))


def day_from_ms(ms: int) -> str:
    return ms_to_dt(ms).strftime("%Y%m%d")


def format_ts(ms: int) -> str:
    return ms_to_dt(ms).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def parse_abs_time(text: str) -> datetime:
    text = text.strip()
    if "." in text:
        return datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f")
    if text.count(":") >= 3:
        head, ms = text.rsplit(":", 1)
        return datetime.strptime(f"{head}.{ms}", "%Y/%m/%d %H:%M:%S.%f")
    return datetime.strptime(text, "%Y/%m/%d %H:%M:%S")


def parse_float(text: str) -> float:
    if text == "":
        return 0.0
    return float(text)


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    return f"{value:.12g}"


def expand_dates(spec: str) -> set[str]:
    dates: set[str] = set()
    if not spec:
        return dates
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cur = datetime.strptime(start, "%Y%m%d")
            end_dt = datetime.strptime(end, "%Y%m%d")
            while cur <= end_dt:
                dates.add(cur.strftime("%Y%m%d"))
                cur += timedelta(days=1)
        else:
            dates.add(part)
    return dates


def iter_days(start: str, end: str) -> Iterable[str]:
    cur = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    while cur <= end_dt:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def read_xlsx_sheet1(path: Path) -> list[list[str]]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as zf:
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root:
                parts = [t.text or "" for t in si.iter(f"{ns}t")]
                shared.append("".join(parts))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(f".//{ns}row"):
            cells: dict[int, str] = {}
            for cell in row.findall(f"{ns}c"):
                ref = cell.attrib.get("r", "")
                idx = column_index(ref)
                value = cell.find(f"{ns}v")
                if value is None:
                    text = ""
                else:
                    text = value.text or ""
                    if cell.attrib.get("t") == "s":
                        text = shared[int(text)]
                cells[idx] = text
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(i, "") for i in range(width)])
        return rows


def load_action_defaults(path: Path) -> dict[int, float]:
    rows = read_xlsx_sheet1(path)
    if not rows:
        raise ValueError(f"empty xlsx: {path}")
    header = [cell.strip() for cell in rows[0]]
    try:
        idx_col = header.index("index")
        default_col = header.index("default")
    except ValueError as exc:
        raise ValueError(f"{path} must contain index/default columns, got {header}") from exc
    defaults: dict[int, float] = {}
    for row in rows[1:]:
        if len(row) <= max(idx_col, default_col) or not row[idx_col]:
            continue
        io_id = int(float(row[idx_col]))
        defaults[io_id] = parse_float(row[default_col])
    return defaults


def read_process_start(path: Path) -> datetime:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("Process Start Time:"):
                return parse_abs_time(line.split(":", 1)[1].strip())
            if line.startswith("Time\t"):
                break
    raise ValueError(f"APC file missing Process Start Time: {path}")


def read_sensor_file(path: Path, source: str, seq_start: int) -> tuple[list[SensorRow], Counter, Counter, int]:
    """Return accepted rows, accepted_by_day, rejected_by_day, next_seq."""
    rows: list[SensorRow] = []
    accepted_by_day: Counter = Counter()
    rejected_by_day: Counter = Counter()
    seq = seq_start
    process_start_ms = dt_to_ms(read_process_start(path)) if source == "apc" else 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if line.startswith("Time\t"):
                header = line.rstrip("\n").split("\t")
                break
        if header is None:
            raise ValueError(f"sensor file missing Time header: {path}")
        time_idx = header.index("Time")
        sensor_idx = [header.index(str(i)) for i in range(1, SENSOR_COUNT + 1)]
        mask_idx = [header.index(f"mask_{i}") for i in range(1, SENSOR_COUNT + 1)]
        max_idx = max(sensor_idx + mask_idx + [time_idx])
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_idx:
                rejected_by_day["unknown"] += 1
                continue
            if source == "apc":
                ts_ms = process_start_ms + int(round(parse_float(parts[time_idx]) * 1000.0))
            else:
                ts_ms = dt_to_ms(parse_abs_time(parts[time_idx]))
            row_day = day_from_ms(ts_ms)
            mask = [parts[i].strip() for i in mask_idx]
            if mask != NORMAL_MASK:
                rejected_by_day[row_day] += 1
                continue
            values = tuple(parse_float(parts[i]) for i in sensor_idx)
            rows.append(SensorRow(ts_ms=ts_ms, seq=seq, values=values))
            seq += 1
            accepted_by_day[row_day] += 1
    return rows, accepted_by_day, rejected_by_day, seq


def read_log_file(path: Path) -> list[tuple[int, int, float, int]]:
    actions: list[tuple[int, int, float, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ts_idx = header.index("timestamp")
        io_idx = header.index("io_id")
        val_idx = header.index("io_value")
        seq = 0
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            ts_ms = dt_to_ms(parse_abs_time(parts[ts_idx]))
            actions.append((ts_ms, int(float(parts[io_idx])), parse_float(parts[val_idx]), seq))
            seq += 1
    return actions


def path_date(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"\d{8}", part):
            return part
    return ""


def candidate_files(dataset: Path, day: str, kind: str) -> list[Path]:
    day_dt = datetime.strptime(day, "%Y%m%d")
    folder_days = [(day_dt + timedelta(days=offset)).strftime("%Y%m%d") for offset in (-1, 0, 1)]
    files: list[Path] = []
    for folder_day in folder_days:
        base = dataset / folder_day
        if kind == "apc":
            files.extend(sorted((base / "sensor").glob("apc_*.txt")))
        elif kind == "livedata":
            files.extend(sorted((base / "sensor").glob("livedata*.txt")))
        elif kind == "log":
            files.extend(sorted((base / "log").glob("*log*.txt")))
    return files


def rows_for_day(rows: list[SensorRow], day: str) -> list[SensorRow]:
    return [row for row in rows if day_from_ms(row.ts_ms) == day]


def actions_for_day(actions: list[tuple[int, int, float, int]], day: str) -> list[tuple[int, int, float, int]]:
    return [action for action in actions if day_from_ms(action[0]) == day]


def make_series(rows: list[SensorRow], gap_ms: int, source_name: str) -> SensorSeries:
    if not rows:
        return SensorSeries([], [], [], gap_ms, source_name)
    rows = sorted(rows, key=lambda r: (r.ts_ms, r.seq))
    dedup: list[SensorRow] = []
    for row in rows:
        if dedup and dedup[-1].ts_ms == row.ts_ms:
            dedup[-1] = row
        else:
            dedup.append(row)
    times = [row.ts_ms for row in dedup]
    values = [row.values for row in dedup]
    seg_ids: list[int] = []
    seg = 0
    prev = None
    for ts in times:
        if prev is not None and ts - prev > gap_ms:
            seg += 1
        seg_ids.append(seg)
        prev = ts
    return SensorSeries(times, values, seg_ids, gap_ms, source_name)


def output_grid_ids(livedata: SensorSeries, day_start: int) -> list[int]:
    grid_ids: list[int] = []
    if not livedata.times:
        return grid_ids
    seg_start = livedata.times[0]
    prev_ts = livedata.times[0]
    prev_seg = livedata.seg_ids[0]
    for ts, seg in zip(livedata.times[1:], livedata.seg_ids[1:]):
        if seg != prev_seg:
            append_segment_grid_ids(grid_ids, seg_start, prev_ts, day_start)
            seg_start = ts
            prev_seg = seg
        prev_ts = ts
    append_segment_grid_ids(grid_ids, seg_start, prev_ts, day_start)
    return sorted(set(gid for gid in grid_ids if gid >= 0))


def append_segment_grid_ids(out: list[int], start_ms: int, end_ms: int, day_start: int) -> None:
    k_start = math.ceil((start_ms - day_start) / GRID_STEP_MS)
    k_end = math.floor((end_ms - day_start) / GRID_STEP_MS)
    if k_end >= k_start:
        out.extend(range(k_start, k_end + 1))


def interpolate(series: SensorSeries, grid_ts: int) -> tuple[tuple[float, ...] | None, bool]:
    if not series.times:
        return None, False
    idx = bisect.bisect_left(series.times, grid_ts)
    if idx < len(series.times) and series.times[idx] == grid_ts:
        return series.values[idx], True
    if idx == 0 or idx >= len(series.times):
        return None, False
    left = idx - 1
    right = idx
    if series.seg_ids[left] != series.seg_ids[right]:
        return None, False
    t_left = series.times[left]
    t_right = series.times[right]
    gap = t_right - t_left
    if gap <= 0 or gap > series.gap_ms:
        return None, False
    if series.source_name == "apc":
        nearest = min(grid_ts - t_left, t_right - grid_ts)
        if nearest > APC_NEAREST_MS:
            return None, False
    ratio = (grid_ts - t_left) / gap
    lv = series.values[left]
    rv = series.values[right]
    return tuple(l + ratio * (r - l) for l, r in zip(lv, rv)), True


def group_actions(actions: list[tuple[int, int, float, int]], day_start: int) -> dict[int, list[tuple[int, int, float, int]]]:
    grouped: dict[int, list[tuple[int, int, float, int]]] = defaultdict(list)
    for ts_ms, io_id, value, seq in sorted(actions, key=lambda x: (x[0], x[3])):
        gid = (ts_ms - day_start) // GRID_STEP_MS
        if gid < 0:
            continue
        grouped[int(gid)].append((ts_ms, io_id, value, seq))
    return dict(grouped)


def apply_actions_for_grid(
    state: list[float],
    grid_actions: list[tuple[int, int, float, int]] | None,
) -> tuple[list[int], int]:
    evt = [0] * (IO_COUNT + 1)
    count = 0
    if not grid_actions:
        return evt, count
    last_by_io: dict[int, tuple[int, int, float]] = {}
    for ts_ms, io_id, value, seq in grid_actions:
        if 1 <= io_id <= IO_COUNT:
            last_by_io[io_id] = (ts_ms, seq, value)
            evt[io_id] = 1
            count += 1
    for io_id, (_, _, value) in last_by_io.items():
        state[io_id] = value
    return evt, count


def read_day_data(dataset: Path, day: str) -> tuple[list[SensorRow], list[SensorRow], list[tuple[int, int, float, int]], dict[str, int]]:
    stats = Counter()
    seq = 0
    apc_rows: list[SensorRow] = []
    livedata_rows: list[SensorRow] = []
    for path in candidate_files(dataset, day, "apc"):
        rows, accepted_by_day, rejected_by_day, seq = read_sensor_file(path, "apc", seq)
        day_rows = rows_for_day(rows, day)
        apc_rows.extend(day_rows)
        stats["accepted_apc_rows"] += len(day_rows)
        stats["rejected_apc_rows"] += rejected_by_day.get(day, 0)
    for path in candidate_files(dataset, day, "livedata"):
        rows, accepted_by_day, rejected_by_day, seq = read_sensor_file(path, "livedata", seq)
        day_rows = rows_for_day(rows, day)
        livedata_rows.extend(day_rows)
        stats["accepted_livedata_rows"] += len(day_rows)
        stats["rejected_livedata_rows"] += rejected_by_day.get(day, 0)
    actions: list[tuple[int, int, float, int]] = []
    for path in candidate_files(dataset, day, "log"):
        actions.extend(actions_for_day(read_log_file(path), day))
    return apc_rows, livedata_rows, actions, dict(stats)


def write_aligned_day(
    out_path: Path,
    day: str,
    apc_series: SensorSeries,
    livedata_series: SensorSeries,
    actions: list[tuple[int, int, float, int]],
    state: list[float],
    last_valid_sensor: list[float] | None,
    overwrite: bool,
) -> tuple[DayStats, list[float] | None]:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; use --overwrite")
    day0 = day_start_ms(day)
    grid_ids = output_grid_ids(livedata_series, day0)
    action_by_grid = group_actions(actions, day0)
    stats = DayStats(date=day, livedata_segments=(max(livedata_series.seg_ids) + 1 if livedata_series.seg_ids else 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        ["timestamp_raw", "ts_ms", "source", "source_code"]
        + [str(i) for i in range(1, SENSOR_COUNT + 1)]
        + [f"mask_{i}" for i in range(1, SENSOR_COUNT + 1)]
        + [f"evt_{i}" for i in range(1, IO_COUNT + 1)]
        + [f"state_{i}" for i in range(1, IO_COUNT + 1)]
        + ["has_action", "action_count"]
    )
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        action_grids = sorted(action_by_grid)
        action_pos = 0
        output_grid_set = set(grid_ids)
        for gid in grid_ids:
            while action_pos < len(action_grids) and action_grids[action_pos] < gid:
                hidden_gid = action_grids[action_pos]
                _, count = apply_actions_for_grid(state, action_by_grid[hidden_gid])
                stats.hidden_actions += count
                action_pos += 1
            evt, action_count = apply_actions_for_grid(state, action_by_grid.get(gid))
            if gid in action_by_grid:
                stats.retained_actions += action_count
                if action_pos < len(action_grids) and action_grids[action_pos] == gid:
                    action_pos += 1

            grid_ts = day0 + gid * GRID_STEP_MS
            sensor, source = choose_sensor(apc_series, livedata_series, grid_ts, last_valid_sensor)
            if source in ("apc_grid", "livedata_grid"):
                last_valid_sensor = list(sensor)
                mask = NORMAL_MASK
            else:
                mask = ALL_ZERO_MASK
            stats.source_counts[source] += 1
            stats.output_rows += 1
            row = (
                [format_ts(grid_ts), str(grid_ts), source, str(SOURCE_CODE[source])]
                + [fmt_float(v) for v in sensor]
                + mask
                + [str(evt[i]) for i in range(1, IO_COUNT + 1)]
                + [fmt_float(state[i]) for i in range(1, IO_COUNT + 1)]
                + [str(1 if action_count else 0), str(action_count)]
            )
            fh.write("\t".join(row) + "\n")

        while action_pos < len(action_grids):
            hidden_gid = action_grids[action_pos]
            if hidden_gid not in output_grid_set:
                _, count = apply_actions_for_grid(state, action_by_grid[hidden_gid])
                stats.hidden_actions += count
            action_pos += 1
    return stats, last_valid_sensor


def choose_sensor(
    apc: SensorSeries,
    livedata: SensorSeries,
    grid_ts: int,
    last_valid_sensor: list[float] | None,
) -> tuple[tuple[float, ...], str]:
    sensor, ok = interpolate(apc, grid_ts)
    if ok and sensor is not None:
        return sensor, "apc_grid"
    sensor, ok = interpolate(livedata, grid_ts)
    if ok and sensor is not None:
        return sensor, "livedata_grid"
    if last_valid_sensor is not None:
        return tuple(last_valid_sensor), "carried_sensor"
    return tuple(0.0 for _ in range(SENSOR_COUNT)), "sensor_default"


def discover_days(dataset: Path) -> list[str]:
    days = set()
    for path in dataset.iterdir():
        if path.is_dir() and re.fullmatch(r"\d{8}", path.name):
            days.add(path.name)
    return sorted(days)


def write_manifest(output: Path, manifest: dict) -> None:
    with (output / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def write_quality_report(output: Path, manifest: dict) -> None:
    lines: list[str] = []
    lines.append("# Alignment Quality Report")
    lines.append("")
    lines.append(f"- dataset: `{manifest['dataset']}`")
    lines.append(f"- output: `{manifest['output']}`")
    lines.append(f"- dates: `{manifest['dates'][0]}` to `{manifest['dates'][-1]}`")
    lines.append(f"- total output rows: `{manifest['totals']['output_rows']}`")
    lines.append(f"- hidden actions: `{manifest['totals']['hidden_actions']}`")
    lines.append(f"- retained actions: `{manifest['totals']['retained_actions']}`")
    lines.append(f"- rejected APC rows: `{manifest['totals']['rejected_apc_rows']}`")
    lines.append(f"- rejected livedata rows: `{manifest['totals']['rejected_livedata_rows']}`")
    lines.append("")
    lines.append("## Source Counts")
    lines.append("")
    lines.append("| source | rows |")
    lines.append("|---|---:|")
    for source, count in sorted(manifest["totals"]["source_counts"].items()):
        lines.append(f"| `{source}` | `{count}` |")
    lines.append("")
    lines.append("## Per Day")
    lines.append("")
    lines.append("| date | rows | livedata_segments | hidden_actions | retained_actions | rejected_apc | rejected_livedata | sources |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for day in manifest["days"]:
        sources = ", ".join(f"{k}:{v}" for k, v in sorted(day["source_counts"].items()))
        lines.append(
            f"| `{day['date']}` | `{day['output_rows']}` | `{day['livedata_segments']}` | "
            f"`{day['hidden_actions']}` | `{day['retained_actions']}` | "
            f"`{day['rejected_apc_rows']}` | `{day['rejected_livedata_rows']}` | `{sources}` |"
        )
    lines.append("")
    (output / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stats_to_dict(stats: DayStats) -> dict:
    return {
        "date": stats.date,
        "livedata_segments": stats.livedata_segments,
        "output_rows": stats.output_rows,
        "hidden_actions": stats.hidden_actions,
        "retained_actions": stats.retained_actions,
        "rejected_apc_rows": stats.rejected_apc_rows,
        "accepted_apc_rows": stats.accepted_apc_rows,
        "rejected_livedata_rows": stats.rejected_livedata_rows,
        "accepted_livedata_rows": stats.accepted_livedata_rows,
        "source_counts": dict(stats.source_counts),
    }


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    aligned_dir = output / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    days = discover_days(dataset)
    if args.start_date:
        days = [day for day in days if day >= args.start_date]
    if args.end_date:
        days = [day for day in days if day <= args.end_date]
    explicit_dates = expand_dates(args.dates)
    if explicit_dates:
        days = [day for day in days if day in explicit_dates]
    if not days:
        raise SystemExit("No dates selected")

    defaults = load_action_defaults(dataset / "action_default.xlsx")
    state = [0.0] * (IO_COUNT + 1)
    for io_id in range(1, IO_COUNT + 1):
        state[io_id] = defaults.get(io_id, 0.0)

    manifest_days: list[dict] = []
    totals = {
        "output_rows": 0,
        "hidden_actions": 0,
        "retained_actions": 0,
        "rejected_apc_rows": 0,
        "accepted_apc_rows": 0,
        "rejected_livedata_rows": 0,
        "accepted_livedata_rows": 0,
        "source_counts": Counter(),
    }
    last_valid_sensor: list[float] | None = None

    for idx, day in enumerate(days, start=1):
        print(f"[{idx}/{len(days)}] reading {day}", flush=True)
        apc_rows, livedata_rows, actions, read_counts = read_day_data(dataset, day)
        apc_series = make_series(apc_rows, APC_GAP_MS, "apc")
        livedata_series = make_series(livedata_rows, LIVEDATA_GAP_MS, "livedata")
        out_path = aligned_dir / f"{day}_aligned.tsv"
        print(
            f"[{idx}/{len(days)}] writing {day}: apc={len(apc_rows)} livedata={len(livedata_rows)} actions={len(actions)}",
            flush=True,
        )
        stats, last_valid_sensor = write_aligned_day(
            out_path,
            day,
            apc_series,
            livedata_series,
            actions,
            state,
            last_valid_sensor,
            args.overwrite,
        )
        stats.accepted_apc_rows = int(read_counts.get("accepted_apc_rows", 0))
        stats.rejected_apc_rows = int(read_counts.get("rejected_apc_rows", 0))
        stats.accepted_livedata_rows = int(read_counts.get("accepted_livedata_rows", 0))
        stats.rejected_livedata_rows = int(read_counts.get("rejected_livedata_rows", 0))
        stats_dict = stats_to_dict(stats)
        manifest_days.append(stats_dict)
        for key in [
            "output_rows",
            "hidden_actions",
            "retained_actions",
            "rejected_apc_rows",
            "accepted_apc_rows",
            "rejected_livedata_rows",
            "accepted_livedata_rows",
        ]:
            totals[key] += stats_dict[key]
        totals["source_counts"].update(stats.source_counts)
        print(
            f"[{idx}/{len(days)}] done {day}: rows={stats.output_rows} hidden_actions={stats.hidden_actions} "
            f"sources={dict(stats.source_counts)}",
            flush=True,
        )

    manifest = {
        "dataset": str(dataset),
        "output": str(output),
        "dates": days,
        "config": {
            "grid_step_ms": GRID_STEP_MS,
            "apc_gap_ms": APC_GAP_MS,
            "apc_nearest_ms": APC_NEAREST_MS,
            "livedata_gap_ms": LIVEDATA_GAP_MS,
            "sensor_count": SENSOR_COUNT,
            "real_sensor_count": REAL_SENSOR_COUNT,
            "io_count": IO_COUNT,
            "normal_mask": "mask_1..148=1, mask_149..150=0",
        },
        "days": manifest_days,
        "totals": {
            **{k: int(v) for k, v in totals.items() if k != "source_counts"},
            "source_counts": dict(totals["source_counts"]),
        },
    }
    write_manifest(output, manifest)
    write_quality_report(output, manifest)
    print(f"Wrote manifest and quality report under {output}", flush=True)


if __name__ == "__main__":
    main()
