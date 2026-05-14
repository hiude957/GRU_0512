#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-sensor accuracy for one continuous GRU closed-loop rollout."
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
    parser.add_argument("--near-zero-threshold", type=float, default=0.005)
    parser.add_argument("--relative-threshold", type=float, default=0.20)
    parser.add_argument("--log-every-steps", type=int, default=100_000)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/eval/gru_0430_replicate_nomask/"
            "msfix_valonly_continuous_per_sensor_accuracy_20260508_20260511.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/eval/gru_0430_replicate_nomask/"
            "msfix_valonly_continuous_per_sensor_accuracy_20260508_20260511.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "outputs/eval/gru_0430_replicate_nomask/"
            "msfix_valonly_continuous_per_sensor_accuracy_20260508_20260511.md"
        ),
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


def run(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    warmup_steps = int(config["warmup_steps"])
    amp = args.amp or str(config.get("amp", "off"))

    start, end, days = selected_range(args.cache, args.split, args.dates)
    rows = end - start
    if rows <= warmup_steps:
        raise ValueError(f"rows={rows} must be greater than warmup_steps={warmup_steps}")

    split_dir = args.cache / args.split
    features_np = np.load(split_dir / "features.npy", mmap_mode="r")
    masks_np = np.load(split_dir / "input_mask.npy", mmap_mode="r")
    segment_features = torch.tensor(
        np.asarray(features_np[start:end], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    segment_masks = torch.tensor(
        np.asarray(masks_np[start:end, :SENSOR_DIM], dtype=np.float32),
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

    correct = torch.zeros(SENSOR_DIM, dtype=torch.float64, device=device)
    valid = torch.zeros(SENSOR_DIM, dtype=torch.float64, device=device)
    abs_error_sum = torch.zeros(SENSOR_DIM, dtype=torch.float64, device=device)

    total_steps = rows - warmup_steps
    started = time.time()
    with torch.no_grad(), autocast_context(device, amp):
        _, hidden = model.gru(segment_features[:warmup_steps].unsqueeze(0))
        for local_idx in range(warmup_steps, rows):
            prediction = model.head(hidden[-1]).squeeze(0).clamp(0.0, 1.0)
            target = segment_features[local_idx, :SENSOR_DIM]
            mask = segment_masks[local_idx]

            sensor_valid = mask > 0
            abs_target = target.abs()
            near_zero = abs_target <= args.near_zero_threshold
            correct_near_zero = prediction.abs() < args.near_zero_threshold
            relative_error = (prediction - target).abs() / abs_target.clamp_min(args.near_zero_threshold)
            correct_nonzero = relative_error < args.relative_threshold
            sensor_correct = torch.where(near_zero, correct_near_zero, correct_nonzero) & sensor_valid

            valid += sensor_valid.to(torch.float64)
            correct += sensor_correct.to(torch.float64)
            abs_error_sum += ((prediction - target).abs() * mask).to(torch.float64)

            next_row = segment_features[local_idx : local_idx + 1].clone()
            next_row[:, :SENSOR_DIM] = prediction
            _, hidden = model.gru(next_row.unsqueeze(0), hidden)

            evaluated = local_idx - warmup_steps + 1
            if args.log_every_steps and evaluated % args.log_every_steps == 0:
                elapsed = time.time() - started
                print(f"rollout step={evaluated}/{total_steps} elapsed_s={elapsed:.1f}", flush=True)

    correct_np = correct.cpu().numpy()
    valid_np = valid.cpu().numpy()
    abs_error_np = abs_error_sum.cpu().numpy()
    rows_out = []
    for sensor_id in range(1, REAL_SENSOR_COUNT + 1):
        idx = sensor_id - 1
        valid_points = int(valid_np[idx])
        correct_points = int(correct_np[idx])
        accuracy = float(correct_np[idx] / valid_np[idx]) if valid_np[idx] else 0.0
        mae = float(abs_error_np[idx] / valid_np[idx]) if valid_np[idx] else 0.0
        rows_out.append(
            {
                "sensor_id": sensor_id,
                "kind": sensor_kind(sensor_id),
                "valid_points": valid_points,
                "correct_points": correct_points,
                "accuracy": accuracy,
                "mae": mae,
            }
        )

    all_valid = int(sum(row["valid_points"] for row in rows_out))
    all_correct = int(sum(row["correct_points"] for row in rows_out))
    all_abs_error = float(sum(row["mae"] * row["valid_points"] for row in rows_out))
    summary = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "split": args.split,
        "dates": [day["date"] for day in days],
        "rollout_policy": "continuous one-chain rollout across selected dates; no segment/day reset",
        "warmup_steps": warmup_steps,
        "evaluated_steps": total_steps,
        "near_zero_threshold": args.near_zero_threshold,
        "relative_threshold": args.relative_threshold,
        "sensor_count": REAL_SENSOR_COUNT,
        "overall_1_148": {
            "valid_points": all_valid,
            "correct_points": all_correct,
            "accuracy": all_correct / all_valid if all_valid else 0.0,
            "mae": all_abs_error / all_valid if all_valid else 0.0,
        },
        "best_sensors": sorted(rows_out, key=lambda row: row["accuracy"], reverse=True)[:10],
        "worst_sensors": sorted(rows_out, key=lambda row: row["accuracy"])[:10],
        "per_sensor": rows_out,
    }
    return summary


def write_outputs(summary: dict, args: argparse.Namespace) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sensor_id", "kind", "valid_points", "correct_points", "accuracy", "mae"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["per_sensor"]:
            writer.writerow(row)

    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Per-Sensor Accuracy - 20260508-20260511 Continuous Rollout",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- cache: `{summary['cache']}`",
        f"- dates: `{', '.join(summary['dates'])}`",
        f"- warmup_steps: `{summary['warmup_steps']}`",
        f"- evaluated_steps: `{summary['evaluated_steps']}`",
        f"- rule: near-zero `< {summary['near_zero_threshold']}`, relative error `< {summary['relative_threshold']}`",
        "",
        "## Overall 1-148",
        "",
        "| valid_points | correct_points | accuracy | mae |",
        "|---:|---:|---:|---:|",
        (
            f"| `{summary['overall_1_148']['valid_points']}` | "
            f"`{summary['overall_1_148']['correct_points']}` | "
            f"`{summary['overall_1_148']['accuracy']:.6f}` | "
            f"`{summary['overall_1_148']['mae']:.6f}` |"
        ),
        "",
        "## Per Sensor",
        "",
        "| sensor_id | kind | valid_points | correct_points | accuracy | mae |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["per_sensor"]:
        lines.append(
            f"| `{row['sensor_id']}` | `{row['kind']}` | `{row['valid_points']}` | "
            f"`{row['correct_points']}` | `{row['accuracy']:.6f}` | `{row['mae']:.6f}` |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = run(args)
    write_outputs(summary, args)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "overall_1_148": summary["overall_1_148"],
                "worst_sensors": summary["worst_sensors"][:5],
                "best_sensors": summary["best_sensors"][:5],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
