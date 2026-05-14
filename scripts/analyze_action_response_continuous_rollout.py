#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train_gru_replicate_0430 import (
    DEFAULT_BINARY_IDS,
    DEFAULT_CONTINUOUS_IDS,
    GRUForecastModel,
    SENSOR_DIM,
    autocast_context,
)


REAL_SENSOR_COUNT = 148
FEATURE_EVT_START = 150
FEATURE_STATE_START = FEATURE_EVT_START + 122


@dataclass(frozen=True)
class Burst:
    start: int
    end: int
    action_rows: int
    action_count: int
    io_ids: tuple[int, ...]
    pre_range: tuple[int, int]
    trend_range: tuple[int, int]
    steady_range: tuple[int, int]


@dataclass(frozen=True)
class Window:
    event_idx: int
    phase: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze post-action trend and steady-state correctness in a continuous GRU rollout."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gru_0430_replicate_nomask/checkpoints/best.pt"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly"),
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--dates", nargs="+", default=["20260508", "20260509", "20260510", "20260511"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default=None)
    parser.add_argument("--burst-gap-ms", type=int, default=2000)
    parser.add_argument("--pre-ms", type=int, default=2000)
    parser.add_argument("--trend-start-ms", type=int, default=1000)
    parser.add_argument("--trend-end-ms", type=int, default=10000)
    parser.add_argument("--steady-start-ms", type=int, default=30000)
    parser.add_argument("--steady-end-ms", type=int, default=60000)
    parser.add_argument("--min-window-rows", type=int, default=3)
    parser.add_argument("--max-window-gap-ms", type=int, default=2000)
    parser.add_argument("--response-threshold", type=float, default=0.005)
    parser.add_argument("--near-zero-threshold", type=float, default=0.005)
    parser.add_argument("--relative-threshold", type=float, default=0.20)
    parser.add_argument("--log-every-steps", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/gru_0430_replicate_nomask/action_response_continuous_20260508_20260511"),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_day_ranges(cache: Path, split: str) -> list[dict]:
    meta = json.loads((cache / split / "meta.json").read_text(encoding="utf-8"))
    return list(meta["day_ranges"])


def selected_range(cache: Path, split: str, dates: list[str]) -> tuple[int, int, list[dict]]:
    requested = set(dates)
    days = [day for day in load_day_ranges(cache, split) if day["date"] in requested and int(day["rows"]) > 0]
    if not days:
        raise ValueError(f"No non-empty days selected: {dates}")
    days = sorted(days, key=lambda item: item["date"])
    return int(days[0]["start"]), int(days[-1]["end"]), days


def sensor_kind(sensor_id: int) -> str:
    if sensor_id in DEFAULT_CONTINUOUS_IDS:
        return "continuous"
    if sensor_id in DEFAULT_BINARY_IDS:
        return "binary"
    return "other"


def contiguous_enough(ts: np.ndarray, start: int, end: int, max_gap_ms: int) -> bool:
    if end - start <= 1:
        return True
    return bool(np.max(np.diff(ts[start:end])) <= max_gap_ms)


def make_range(ts: np.ndarray, start_ms: int, end_ms: int) -> tuple[int, int]:
    return int(np.searchsorted(ts, start_ms, side="left")), int(np.searchsorted(ts, end_ms, side="right"))


def build_bursts(
    ts: np.ndarray,
    features: np.ndarray,
    has_action: np.ndarray,
    action_count: np.ndarray,
    warmup_steps: int,
    args: argparse.Namespace,
) -> tuple[list[Burst], dict]:
    action_idx = np.flatnonzero(has_action)
    raw_bursts: list[tuple[int, int]] = []
    if len(action_idx):
        start = end = int(action_idx[0])
        for cur in action_idx[1:]:
            cur = int(cur)
            if int(ts[cur]) - int(ts[end]) <= args.burst_gap_ms:
                end = cur
            else:
                raw_bursts.append((start, end))
                start = end = cur
        raw_bursts.append((start, end))

    selected: list[Burst] = []
    reject_reasons = {
        "before_warmup": 0,
        "next_action_too_close": 0,
        "short_window": 0,
        "window_has_large_gap": 0,
    }
    for idx, (start, end) in enumerate(raw_bursts):
        next_start = raw_bursts[idx + 1][0] if idx + 1 < len(raw_bursts) else len(ts) - 1
        if start - warmup_steps < 1:
            reject_reasons["before_warmup"] += 1
            continue
        if int(ts[next_start]) - int(ts[end]) < args.steady_end_ms:
            reject_reasons["next_action_too_close"] += 1
            continue

        pre_range = make_range(ts, int(ts[start]) - args.pre_ms, int(ts[start]) - 1)
        trend_range = make_range(ts, int(ts[end]) + args.trend_start_ms, int(ts[end]) + args.trend_end_ms)
        steady_range = make_range(ts, int(ts[end]) + args.steady_start_ms, int(ts[end]) + args.steady_end_ms)
        ranges = [pre_range, trend_range, steady_range]
        if any(r_end - r_start < args.min_window_rows for r_start, r_end in ranges):
            reject_reasons["short_window"] += 1
            continue
        if pre_range[0] < warmup_steps:
            reject_reasons["before_warmup"] += 1
            continue
        if any(not contiguous_enough(ts, r_start, r_end, args.max_window_gap_ms) for r_start, r_end in ranges):
            reject_reasons["window_has_large_gap"] += 1
            continue

        evt = features[start : end + 1, FEATURE_EVT_START:FEATURE_STATE_START]
        io_ids = tuple(int(i + 1) for i in np.flatnonzero(evt.max(axis=0) > 0))
        selected.append(
            Burst(
                start=start,
                end=end,
                action_rows=end - start + 1,
                action_count=int(action_count[start : end + 1].sum()),
                io_ids=io_ids,
                pre_range=pre_range,
                trend_range=trend_range,
                steady_range=steady_range,
            )
        )

    return selected, {
        "action_rows": int(len(action_idx)),
        "action_count": int(action_count[action_idx].sum()) if len(action_idx) else 0,
        "raw_bursts": len(raw_bursts),
        "selected_bursts": len(selected),
        "reject_reasons": reject_reasons,
    }


def sign_with_deadband(values: np.ndarray, threshold: float) -> np.ndarray:
    signs = np.zeros_like(values, dtype=np.int8)
    signs[values > threshold] = 1
    signs[values < -threshold] = -1
    return signs


def steady_value_correct(pred: np.ndarray, true: np.ndarray, near_zero: float, relative: float) -> np.ndarray:
    abs_true = np.abs(true)
    near = abs_true <= near_zero
    correct_near = np.abs(pred) < near_zero
    correct_rel = np.abs(pred - true) / np.maximum(abs_true, near_zero) < relative
    return np.where(near, correct_near, correct_rel)


def run_rollout_collect_windows(
    args: argparse.Namespace,
    features_np: np.ndarray,
    ts: np.ndarray,
    bursts: list[Burst],
    warmup_steps: int,
    checkpoint: dict,
    device: torch.device,
    amp: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    phase_ranges = {
        "pre": [burst.pre_range for burst in bursts],
        "trend": [burst.trend_range for burst in bursts],
        "steady": [burst.steady_range for burst in bursts],
    }
    windows: list[Window] = []
    for event_idx, burst in enumerate(bursts):
        windows.extend(
            [
                Window(event_idx, "pre", burst.pre_range[0], burst.pre_range[1]),
                Window(event_idx, "trend", burst.trend_range[0], burst.trend_range[1]),
                Window(event_idx, "steady", burst.steady_range[0], burst.steady_range[1]),
            ]
        )
    windows = sorted(windows, key=lambda item: item.start)
    event_count = len(bursts)
    sums_pred = {phase: np.zeros((event_count, REAL_SENSOR_COUNT), dtype=np.float64) for phase in phase_ranges}
    sums_true = {phase: np.zeros((event_count, REAL_SENSOR_COUNT), dtype=np.float64) for phase in phase_ranges}
    counts = {phase: np.zeros(event_count, dtype=np.int64) for phase in phase_ranges}

    segment_features = torch.tensor(
        np.asarray(features_np, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    model = GRUForecastModel(
        input_dim=int(segment_features.shape[1]),
        output_dim=SENSOR_DIM,
        hidden_size=int(checkpoint["config"]["hidden_size"]),
        num_layers=int(checkpoint["config"]["num_layers"]),
        dropout=float(checkpoint["config"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    active: list[Window] = []
    start_ptr = 0
    total_steps = len(ts) - warmup_steps
    started = time.time()
    with torch.no_grad(), autocast_context(device, amp):
        _, hidden = model.gru(segment_features[:warmup_steps].unsqueeze(0))
        for local_idx in range(warmup_steps, len(ts)):
            while start_ptr < len(windows) and windows[start_ptr].start <= local_idx:
                if windows[start_ptr].end > local_idx:
                    active.append(windows[start_ptr])
                start_ptr += 1
            if active:
                active = [window for window in active if window.end > local_idx]

            prediction = model.head(hidden[-1]).squeeze(0).clamp(0.0, 1.0)
            if active:
                pred_np = prediction[:REAL_SENSOR_COUNT].float().cpu().numpy().astype(np.float64, copy=False)
                true_np = segment_features[local_idx, :REAL_SENSOR_COUNT].float().cpu().numpy().astype(np.float64, copy=False)
                for window in active:
                    sums_pred[window.phase][window.event_idx] += pred_np
                    sums_true[window.phase][window.event_idx] += true_np
                    counts[window.phase][window.event_idx] += 1

            next_row = segment_features[local_idx : local_idx + 1].clone()
            next_row[:, :SENSOR_DIM] = prediction
            _, hidden = model.gru(next_row.unsqueeze(0), hidden)

            evaluated = local_idx - warmup_steps + 1
            if args.log_every_steps and evaluated % args.log_every_steps == 0:
                elapsed = time.time() - started
                print(f"rollout step={evaluated}/{total_steps} elapsed_s={elapsed:.1f}", flush=True)

    means_pred = {phase: sums_pred[phase] / counts[phase][:, None] for phase in sums_pred}
    means_true = {phase: sums_true[phase] / counts[phase][:, None] for phase in sums_true}
    return means_pred, means_true


def summarize(
    args: argparse.Namespace,
    bursts: list[Burst],
    burst_stats: dict,
    means_pred: dict[str, np.ndarray],
    means_true: dict[str, np.ndarray],
    days: list[dict],
    warmup_steps: int,
) -> dict:
    pre_true = means_true["pre"]
    pre_pred = means_pred["pre"]
    trend_true = means_true["trend"]
    trend_pred = means_pred["trend"]
    steady_true = means_true["steady"]
    steady_pred = means_pred["steady"]

    trend_delta_true = trend_true - pre_true
    trend_delta_pred = trend_pred - pre_pred
    steady_delta_true = steady_true - pre_true
    steady_delta_pred = steady_pred - pre_pred

    trend_changed = np.abs(trend_delta_true) >= args.response_threshold
    steady_changed = np.abs(steady_delta_true) >= args.response_threshold
    trend_dir_correct = sign_with_deadband(trend_delta_true, args.response_threshold) == sign_with_deadband(
        trend_delta_pred, args.response_threshold
    )
    steady_dir_correct = sign_with_deadband(steady_delta_true, args.response_threshold) == sign_with_deadband(
        steady_delta_pred, args.response_threshold
    )
    steady_value_ok = steady_value_correct(
        steady_pred, steady_true, args.near_zero_threshold, args.relative_threshold
    )
    steady_abs_error = np.abs(steady_pred - steady_true)
    steady_delta_abs_error = np.abs(steady_delta_pred - steady_delta_true)

    per_sensor: list[dict] = []
    for sensor_id in range(1, REAL_SENSOR_COUNT + 1):
        idx = sensor_id - 1
        trend_mask = trend_changed[:, idx]
        steady_mask = steady_changed[:, idx]
        trend_count = int(trend_mask.sum())
        steady_count = int(steady_mask.sum())
        trend_correct_count = int((trend_dir_correct[:, idx] & trend_mask).sum())
        steady_dir_correct_count = int((steady_dir_correct[:, idx] & steady_mask).sum())
        steady_value_all_count = int(steady_value_ok[:, idx].sum())
        steady_value_changed_count = int((steady_value_ok[:, idx] & steady_mask).sum())
        ratio_values = np.abs(steady_delta_pred[:, idx][steady_mask]) / np.maximum(
            np.abs(steady_delta_true[:, idx][steady_mask]), args.response_threshold
        )
        per_sensor.append(
            {
                "sensor_id": sensor_id,
                "kind": sensor_kind(sensor_id),
                "events": len(bursts),
                "trend_changed_events": trend_count,
                "trend_direction_correct": trend_correct_count,
                "trend_direction_accuracy": trend_correct_count / trend_count if trend_count else None,
                "steady_changed_events": steady_count,
                "steady_direction_correct": steady_dir_correct_count,
                "steady_direction_accuracy": steady_dir_correct_count / steady_count if steady_count else None,
                "steady_value_correct_all": steady_value_all_count,
                "steady_value_accuracy_all": steady_value_all_count / len(bursts) if bursts else None,
                "steady_value_correct_changed": steady_value_changed_count,
                "steady_value_accuracy_changed": steady_value_changed_count / steady_count if steady_count else None,
                "steady_mae": float(steady_abs_error[:, idx].mean()),
                "steady_bias": float((steady_pred[:, idx] - steady_true[:, idx]).mean()),
                "steady_delta_mae_changed": float(steady_delta_abs_error[:, idx][steady_mask].mean())
                if steady_count
                else None,
                "median_abs_delta_ratio_changed": float(statistics.median(ratio_values.tolist()))
                if len(ratio_values)
                else None,
            }
        )

    changed_pairs = steady_changed
    all_steady_value_acc = float(steady_value_ok.mean()) if len(bursts) else 0.0
    changed_steady_value_acc = float((steady_value_ok & changed_pairs).sum() / changed_pairs.sum()) if changed_pairs.sum() else 0.0
    trend_dir_acc = float((trend_dir_correct & trend_changed).sum() / trend_changed.sum()) if trend_changed.sum() else 0.0
    steady_dir_acc = float((steady_dir_correct & steady_changed).sum() / steady_changed.sum()) if steady_changed.sum() else 0.0

    event_rows: list[dict] = []
    for event_idx, burst in enumerate(bursts):
        steady_changed_count = int(steady_changed[event_idx].sum())
        trend_changed_count = int(trend_changed[event_idx].sum())
        event_rows.append(
            {
                "event_id": event_idx + 1,
                "start_idx": burst.start,
                "end_idx": burst.end,
                "action_rows": burst.action_rows,
                "action_count": burst.action_count,
                "changed_io_count": len(burst.io_ids),
                "io_ids": ",".join(str(io_id) for io_id in burst.io_ids[:30]),
                "trend_changed_sensors": trend_changed_count,
                "trend_direction_accuracy": float(
                    (trend_dir_correct[event_idx] & trend_changed[event_idx]).sum() / trend_changed_count
                )
                if trend_changed_count
                else None,
                "steady_changed_sensors": steady_changed_count,
                "steady_direction_accuracy": float(
                    (steady_dir_correct[event_idx] & steady_changed[event_idx]).sum() / steady_changed_count
                )
                if steady_changed_count
                else None,
                "steady_value_accuracy_all": float(steady_value_ok[event_idx].mean()),
                "steady_mae": float(steady_abs_error[event_idx].mean()),
            }
        )

    return {
        "config": {
            "dates": [day["date"] for day in days],
            "warmup_steps": warmup_steps,
            "burst_gap_ms": args.burst_gap_ms,
            "pre_ms": args.pre_ms,
            "trend_window_ms": [args.trend_start_ms, args.trend_end_ms],
            "steady_window_ms": [args.steady_start_ms, args.steady_end_ms],
            "response_threshold": args.response_threshold,
            "near_zero_threshold": args.near_zero_threshold,
            "relative_threshold": args.relative_threshold,
        },
        "burst_stats": burst_stats,
        "overall": {
            "events": len(bursts),
            "sensor_count": REAL_SENSOR_COUNT,
            "trend_changed_pairs": int(trend_changed.sum()),
            "trend_direction_accuracy_changed": trend_dir_acc,
            "steady_changed_pairs": int(steady_changed.sum()),
            "steady_direction_accuracy_changed": steady_dir_acc,
            "steady_value_accuracy_all_pairs": all_steady_value_acc,
            "steady_value_accuracy_changed_pairs": changed_steady_value_acc,
            "steady_mae_all_pairs": float(steady_abs_error.mean()) if len(bursts) else 0.0,
        },
        "per_sensor": per_sensor,
        "per_event": event_rows,
    }


def fmt_optional(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sensor_fields = list(summary["per_sensor"][0].keys())
    with (output_dir / "per_sensor_action_response.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sensor_fields)
        writer.writeheader()
        writer.writerows(summary["per_sensor"])

    event_fields = list(summary["per_event"][0].keys()) if summary["per_event"] else []
    with (output_dir / "per_event_action_response.csv").open("w", encoding="utf-8", newline="") as fh:
        if event_fields:
            writer = csv.DictWriter(fh, fieldnames=event_fields)
            writer.writeheader()
            writer.writerows(summary["per_event"])

    worst_trend = sorted(
        [row for row in summary["per_sensor"] if row["trend_direction_accuracy"] is not None],
        key=lambda row: row["trend_direction_accuracy"],
    )[:10]
    worst_steady = sorted(
        [row for row in summary["per_sensor"] if row["steady_value_accuracy_changed"] is not None],
        key=lambda row: row["steady_value_accuracy_changed"],
    )[:10]
    best_steady = sorted(
        [row for row in summary["per_sensor"] if row["steady_value_accuracy_changed"] is not None],
        key=lambda row: row["steady_value_accuracy_changed"],
        reverse=True,
    )[:10]
    lines = [
        "# Action Response Analysis - Continuous Rollout",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary["overall"].items():
        lines.append(f"| `{key}` | `{fmt_optional(value)}` |")
    lines.extend(
        [
            "",
            "## Worst Trend Direction Sensors",
            "",
            "| sensor | kind | changed_events | trend_dir_acc | steady_value_acc_changed | steady_mae |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in worst_trend:
        lines.append(
            f"| `{row['sensor_id']}` | `{row['kind']}` | `{row['trend_changed_events']}` | "
            f"`{fmt_optional(row['trend_direction_accuracy'])}` | "
            f"`{fmt_optional(row['steady_value_accuracy_changed'])}` | `{row['steady_mae']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Worst Steady Value Sensors",
            "",
            "| sensor | kind | steady_changed_events | steady_value_acc_changed | steady_dir_acc | steady_mae | bias |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst_steady:
        lines.append(
            f"| `{row['sensor_id']}` | `{row['kind']}` | `{row['steady_changed_events']}` | "
            f"`{fmt_optional(row['steady_value_accuracy_changed'])}` | "
            f"`{fmt_optional(row['steady_direction_accuracy'])}` | `{row['steady_mae']:.6f}` | "
            f"`{row['steady_bias']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Best Steady Value Sensors",
            "",
            "| sensor | kind | steady_changed_events | steady_value_acc_changed | steady_dir_acc | steady_mae | bias |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_steady:
        lines.append(
            f"| `{row['sensor_id']}` | `{row['kind']}` | `{row['steady_changed_events']}` | "
            f"`{fmt_optional(row['steady_value_accuracy_changed'])}` | "
            f"`{fmt_optional(row['steady_direction_accuracy'])}` | `{row['steady_mae']:.6f}` | "
            f"`{row['steady_bias']:.6f}` |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    warmup_steps = int(checkpoint["config"]["warmup_steps"])
    amp = args.amp or str(checkpoint["config"].get("amp", "off"))

    split_dir = args.cache / args.split
    features_all = np.load(split_dir / "features.npy", mmap_mode="r")
    ts_all = np.load(split_dir / "ts_ms.npy", mmap_mode="r")
    has_all = np.load(split_dir / "has_action.npy", mmap_mode="r")
    action_count_all = np.load(split_dir / "action_count.npy", mmap_mode="r")
    start, end, days = selected_range(args.cache, args.split, args.dates)
    features = np.asarray(features_all[start:end], dtype=np.float32)
    ts = np.asarray(ts_all[start:end], dtype=np.int64)
    has_action = np.asarray(has_all[start:end], dtype=np.bool_)
    action_count = np.asarray(action_count_all[start:end], dtype=np.int16)

    bursts, burst_stats = build_bursts(ts, features, has_action, action_count, warmup_steps, args)
    print(json.dumps({"burst_stats": burst_stats}, indent=2), flush=True)
    if not bursts:
        raise SystemExit("No action bursts selected for analysis")

    means_pred, means_true = run_rollout_collect_windows(
        args=args,
        features_np=features,
        ts=ts,
        bursts=bursts,
        warmup_steps=warmup_steps,
        checkpoint=checkpoint,
        device=device,
        amp=amp,
    )
    summary = summarize(args, bursts, burst_stats, means_pred, means_true, days, warmup_steps)
    write_outputs(summary, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "overall": summary["overall"],
                "burst_stats": burst_stats,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
