#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_gru_replicate_0430 import GRUForecastModel, SENSOR_DIM, autocast_context


EPOCH = datetime(1970, 1, 1)
REAL_SENSOR_COUNT = 148


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pred/true time-domain curves for continuous closed-loop GRU rollout."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gru_0430_replicate_nomask/checkpoints/best.pt"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly"),
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--dates", nargs="+", default=["20260508", "20260509", "20260510", "20260511"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots/gru_continuous_20260508_20260511"),
    )
    parser.add_argument("--sample-count", type=int, default=12_000)
    parser.add_argument("--log-every-steps", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default=None)
    parser.add_argument("--gap-break-ms", type=int, default=30 * 60 * 1000)
    return parser.parse_args()


def dt_to_ms(dt: datetime) -> int:
    delta = dt - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000) + (delta.microseconds // 1000)


def ms_to_dt(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=int(ms))


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


def build_sample_indices(warmup_steps: int, rows: int, sample_count: int) -> np.ndarray:
    if rows <= warmup_steps:
        raise ValueError(f"rows={rows} must be greater than warmup_steps={warmup_steps}")
    count = min(sample_count, rows - warmup_steps)
    return np.unique(np.linspace(warmup_steps, rows - 1, count, dtype=np.int64))


def add_gap_breaks(x: np.ndarray, y: np.ndarray, ts_ms: np.ndarray, gap_break_ms: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= 1:
        return x, y
    break_after = np.flatnonzero(np.diff(ts_ms) > gap_break_ms)
    if len(break_after) == 0:
        return x, y
    pieces_x: list[np.ndarray] = []
    pieces_y: list[np.ndarray] = []
    start = 0
    for idx in break_after:
        stop = int(idx) + 1
        pieces_x.append(x[start:stop])
        pieces_y.append(y[start:stop])
        pieces_x.append(np.array([np.nan], dtype=x.dtype))
        pieces_y.append(np.array([np.nan], dtype=y.dtype))
        start = stop
    pieces_x.append(x[start:])
    pieces_y.append(y[start:])
    return np.concatenate(pieces_x), np.concatenate(pieces_y)


def plot_single_sensor(
    out_path: Path,
    x_hours: np.ndarray,
    ts_ms: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    sensor_id: int,
    dates: list[str],
    gap_break_ms: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2), dpi=160)
    tx, ty = add_gap_breaks(x_hours, true_values, ts_ms, gap_break_ms)
    px, py = add_gap_breaks(x_hours, pred_values, ts_ms, gap_break_ms)
    ax.plot(tx, ty, color="#2563eb", linewidth=0.8, label="true", alpha=0.85)
    ax.plot(px, py, color="#f97316", linewidth=0.8, label="pred", alpha=0.78)
    for day in dates[1:]:
        boundary = (dt_to_ms(datetime.strptime(day, "%Y%m%d")) - int(ts_ms[0])) / 3_600_000.0
        if 0 <= boundary <= x_hours[-1]:
            ax.axvline(boundary, color="#64748b", linewidth=0.6, linestyle="--", alpha=0.45)
            ax.text(boundary, 1.02, day, transform=ax.get_xaxis_transform(), fontsize=7, color="#475569")
    ax.set_title(f"Sensor {sensor_id:03d}: pred vs true | continuous autoregressive rollout", fontsize=11)
    ax.set_xlabel("Elapsed hours from first plotted prediction")
    ax.set_ylabel("Normalized sensor value")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, color="#d8dee9", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_page(
    out_path: Path,
    x_hours: np.ndarray,
    ts_ms: np.ndarray,
    true_samples: np.ndarray,
    pred_samples: np.ndarray,
    sensor_ids: list[int],
    dates: list[str],
    gap_break_ms: int,
) -> None:
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10), dpi=160, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, sensor_id in zip(axes_flat, sensor_ids):
        idx = sensor_id - 1
        tx, ty = add_gap_breaks(x_hours, true_samples[:, idx], ts_ms, gap_break_ms)
        px, py = add_gap_breaks(x_hours, pred_samples[:, idx], ts_ms, gap_break_ms)
        ax.plot(tx, ty, color="#2563eb", linewidth=0.55, alpha=0.85)
        ax.plot(px, py, color="#f97316", linewidth=0.55, alpha=0.78)
        for day in dates[1:]:
            boundary = (dt_to_ms(datetime.strptime(day, "%Y%m%d")) - int(ts_ms[0])) / 3_600_000.0
            if 0 <= boundary <= x_hours[-1]:
                ax.axvline(boundary, color="#64748b", linewidth=0.45, linestyle="--", alpha=0.35)
        ax.set_title(f"S{sensor_id:03d}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, color="#d8dee9", linewidth=0.35, alpha=0.65)
    for ax in axes_flat[len(sensor_ids):]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#2563eb", linewidth=1.2, label="true"),
        plt.Line2D([0], [0], color="#f97316", linewidth=1.2, label="pred"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Continuous autoregressive rollout pred vs true | {dates[0]}-{dates[-1]} | sensors {sensor_ids[0]}-{sensor_ids[-1]}",
        fontsize=13,
        y=0.985,
    )
    fig.supxlabel("Elapsed hours from first plotted prediction", fontsize=10)
    fig.supylabel("Normalized sensor value", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def run_rollout_samples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    warmup_steps = int(config["warmup_steps"])
    amp = args.amp or str(config.get("amp", "off"))

    split_dir = args.cache / args.split
    features_np = np.load(split_dir / "features.npy", mmap_mode="r")
    ts_np = np.load(split_dir / "ts_ms.npy", mmap_mode="r")
    start, end, days = selected_range(args.cache, args.split, args.dates)
    rows = end - start
    sample_local_indices = build_sample_indices(warmup_steps, rows, args.sample_count)
    sample_lookup = {int(local_idx): pos for pos, local_idx in enumerate(sample_local_indices)}
    sample_count = len(sample_local_indices)

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "cache": str(args.cache),
                "dates": [day["date"] for day in days],
                "rows": rows,
                "warmup_steps": warmup_steps,
                "sample_count": sample_count,
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )

    segment_features = torch.tensor(
        np.asarray(features_np[start:end], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    model = GRUForecastModel(
        input_dim=int(segment_features.shape[1]),
        output_dim=SENSOR_DIM,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_samples = np.empty((sample_count, REAL_SENSOR_COUNT), dtype=np.float32)
    true_samples = np.empty((sample_count, REAL_SENSOR_COUNT), dtype=np.float32)
    ts_samples = np.asarray(ts_np[start + sample_local_indices], dtype=np.int64)

    total_steps = rows - warmup_steps
    started = time.time()
    with torch.no_grad(), autocast_context(device, amp):
        _, hidden = model.gru(segment_features[:warmup_steps].unsqueeze(0))
        for local_idx in range(warmup_steps, rows):
            prediction = model.head(hidden[-1]).squeeze(0).clamp(0.0, 1.0)
            sample_pos = sample_lookup.get(local_idx)
            if sample_pos is not None:
                pred_samples[sample_pos] = prediction[:REAL_SENSOR_COUNT].float().cpu().numpy()
                true_samples[sample_pos] = (
                    segment_features[local_idx, :REAL_SENSOR_COUNT].float().cpu().numpy()
                )

            next_row = segment_features[local_idx : local_idx + 1].clone()
            next_row[:, :SENSOR_DIM] = prediction
            _, hidden = model.gru(next_row.unsqueeze(0), hidden)

            evaluated = local_idx - warmup_steps + 1
            if args.log_every_steps and evaluated % args.log_every_steps == 0:
                elapsed = time.time() - started
                print(f"rollout step={evaluated}/{total_steps} elapsed_s={elapsed:.1f}", flush=True)

    meta = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "split": args.split,
        "dates": [day["date"] for day in days],
        "range": {"start": start, "end": end, "rows": rows},
        "warmup_steps": warmup_steps,
        "sample_count": sample_count,
        "first_sample_ts": int(ts_samples[0]),
        "last_sample_ts": int(ts_samples[-1]),
        "first_sample_time": ms_to_dt(int(ts_samples[0])).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "last_sample_time": ms_to_dt(int(ts_samples[-1])).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "sensor_ids": list(range(1, REAL_SENSOR_COUNT + 1)),
        "gap_break_ms": args.gap_break_ms,
    }
    return ts_samples, true_samples, pred_samples, meta


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page_dir = args.output_dir / "pages"
    sensor_dir = args.output_dir / "per_sensor"
    page_dir.mkdir(parents=True, exist_ok=True)
    sensor_dir.mkdir(parents=True, exist_ok=True)

    ts_samples, true_samples, pred_samples, meta = run_rollout_samples(args)
    x_hours = (ts_samples - int(ts_samples[0])) / 3_600_000.0

    page_paths: list[str] = []
    for page_idx, start_sensor in enumerate(range(1, REAL_SENSOR_COUNT + 1, 12), start=1):
        sensor_ids = list(range(start_sensor, min(start_sensor + 12, REAL_SENSOR_COUNT + 1)))
        out_path = page_dir / f"page_{page_idx:02d}_s{sensor_ids[0]:03d}_s{sensor_ids[-1]:03d}.png"
        plot_page(out_path, x_hours, ts_samples, true_samples, pred_samples, sensor_ids, args.dates, args.gap_break_ms)
        page_paths.append(str(out_path))
        print(f"wrote {out_path}", flush=True)

    sensor_paths: list[str] = []
    for sensor_id in range(1, REAL_SENSOR_COUNT + 1):
        out_path = sensor_dir / f"sensor_{sensor_id:03d}.png"
        plot_single_sensor(
            out_path,
            x_hours,
            ts_samples,
            true_samples[:, sensor_id - 1],
            pred_samples[:, sensor_id - 1],
            sensor_id,
            args.dates,
            args.gap_break_ms,
        )
        sensor_paths.append(str(out_path))
        if sensor_id % 20 == 0:
            print(f"plotted {sensor_id}/{REAL_SENSOR_COUNT} sensors", flush=True)

    meta["page_plots"] = page_paths
    meta["per_sensor_dir"] = str(sensor_dir)
    meta["per_sensor_count"] = len(sensor_paths)
    (args.output_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "pages": len(page_paths), "per_sensor": len(sensor_paths)}, indent=2))


if __name__ == "__main__":
    main()
