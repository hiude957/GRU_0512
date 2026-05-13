#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from train_gru_replicate_0430 import (
    DEFAULT_BINARY_IDS,
    DEFAULT_CONTINUOUS_IDS,
    SENSOR_DIM,
    GRUForecastModel,
    autocast_context,
    id_to_indices,
)


@dataclass
class MetricAccumulator:
    correct: int = 0
    valid: int = 0
    abs_error_sum: float = 0.0

    def update(self, correct: int, valid: int, abs_error_sum: float) -> None:
        self.correct += correct
        self.valid += valid
        self.abs_error_sum += abs_error_sum

    @property
    def accuracy(self) -> float:
        return self.correct / self.valid if self.valid else 0.0

    @property
    def mae(self) -> float:
        return self.abs_error_sum / self.valid if self.valid else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run day/segment-level closed-loop autoregressive evaluation for GRU replicate checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/gru_0430_replicate_nomask/checkpoints/best.pt"),
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-gap-ms", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default=None)
    parser.add_argument("--near-zero-threshold", type=float, default=0.005)
    parser.add_argument("--relative-threshold", type=float, default=0.20)
    parser.add_argument("--log-every-steps", type=int, default=50000)
    parser.add_argument("--max-steps-per-segment", type=int, default=0)
    parser.add_argument(
        "--rollout-mode",
        choices=("segments", "day", "continuous"),
        default="segments",
        help=(
            "segments: reset at gaps > max-gap-ms; "
            "day: one autoregressive chain per day; "
            "continuous: one chain across all selected days"
        ),
    )
    parser.add_argument("--quiet-segments", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval/gru_0430_replicate_nomask/val_closed_loop_days_best.json"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_day_ranges(cache: Path, split: str) -> list[dict]:
    meta = json.loads((cache / split / "meta.json").read_text(encoding="utf-8"))
    return list(meta["day_ranges"])


def split_segments(ts_ms: np.ndarray, start: int, end: int, max_gap_ms: int) -> list[tuple[int, int]]:
    if end <= start:
        return []
    day_ts = ts_ms[start:end]
    breaks = np.flatnonzero(np.diff(day_ts) > max_gap_ms) + 1
    starts = np.concatenate(([0], breaks)).astype(np.int64)
    ends = np.concatenate((breaks, [len(day_ts)])).astype(np.int64)
    return [(start + int(seg_start), start + int(seg_end)) for seg_start, seg_end in zip(starts, ends)]


def build_eval_groups(
    selected_days: list[dict],
    ts_ms: np.ndarray,
    max_gap_ms: int,
    rollout_mode: str,
) -> list[tuple[str, list[tuple[int, int]]]]:
    if rollout_mode == "continuous":
        if not selected_days:
            return []
        label = f"{selected_days[0]['date']}-{selected_days[-1]['date']}"
        return [(label, [(int(selected_days[0]["start"]), int(selected_days[-1]["end"]))])]

    groups: list[tuple[str, list[tuple[int, int]]]] = []
    for day in selected_days:
        start = int(day["start"])
        end = int(day["end"])
        if rollout_mode == "day":
            segments = [(start, end)] if end > start else []
        else:
            segments = split_segments(ts_ms, start, end, max_gap_ms)
        groups.append((str(day["date"]), segments))
    return groups


def reset_policy_text(rollout_mode: str, max_gap_ms: int) -> str:
    if rollout_mode == "continuous":
        return "single autoregressive chain across selected days; no reset at segment or day boundaries"
    if rollout_mode == "day":
        return "reset once per selected day; no reset at segment gaps inside a day"
    return f"reset at each date and at ts_ms gaps greater than {max_gap_ms} ms"


def compute_metric_chunk(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    near_zero_threshold: float,
    relative_threshold: float,
) -> tuple[int, int, float]:
    valid = masks > 0
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return 0, 0, 0.0
    abs_target = targets.abs()
    near_zero = abs_target <= near_zero_threshold
    correct_near_zero = predictions.abs() < near_zero_threshold
    relative_error = (predictions - targets).abs() / abs_target.clamp_min(near_zero_threshold)
    correct_nonzero = relative_error < relative_threshold
    correct = torch.where(near_zero, correct_near_zero, correct_nonzero) & valid
    abs_error_sum = float(((predictions - targets).abs() * masks).sum().item())
    return int(correct.sum().item()), valid_count, abs_error_sum


def flush_metric_buffers(
    prediction_buffer: list[torch.Tensor],
    target_buffer: list[torch.Tensor],
    mask_buffer: list[torch.Tensor],
    continuous_indices: torch.Tensor,
    binary_indices: torch.Tensor,
    near_zero_threshold: float,
    relative_threshold: float,
    all_metrics: MetricAccumulator,
    continuous_metrics: MetricAccumulator,
    binary_metrics: MetricAccumulator,
) -> None:
    if not prediction_buffer:
        return
    predictions = torch.stack(prediction_buffer, dim=0)
    targets = torch.stack(target_buffer, dim=0)
    masks = torch.stack(mask_buffer, dim=0)

    correct, valid, abs_error_sum = compute_metric_chunk(
        predictions, targets, masks, near_zero_threshold, relative_threshold
    )
    all_metrics.update(correct, valid, abs_error_sum)

    correct, valid, abs_error_sum = compute_metric_chunk(
        predictions.index_select(1, continuous_indices),
        targets.index_select(1, continuous_indices),
        masks.index_select(1, continuous_indices),
        near_zero_threshold,
        relative_threshold,
    )
    continuous_metrics.update(correct, valid, abs_error_sum)

    correct, valid, abs_error_sum = compute_metric_chunk(
        predictions.index_select(1, binary_indices),
        targets.index_select(1, binary_indices),
        masks.index_select(1, binary_indices),
        near_zero_threshold,
        relative_threshold,
    )
    binary_metrics.update(correct, valid, abs_error_sum)
    prediction_buffer.clear()
    target_buffer.clear()
    mask_buffer.clear()


def evaluate_segment(
    model: GRUForecastModel,
    features_np: np.ndarray,
    masks_np: np.ndarray,
    segment_start: int,
    segment_end: int,
    warmup_steps: int,
    device: torch.device,
    amp: str,
    continuous_indices: torch.Tensor,
    binary_indices: torch.Tensor,
    near_zero_threshold: float,
    relative_threshold: float,
    log_every_steps: int,
    max_steps_per_segment: int,
    progress: tqdm | None,
) -> dict:
    segment_rows = segment_end - segment_start
    if segment_rows <= warmup_steps:
        return {
            "start": segment_start,
            "end": segment_end,
            "rows": segment_rows,
            "warmup_steps": warmup_steps,
            "evaluated_steps": 0,
            "skipped": True,
            "reason": "segment shorter than warmup_steps + 1",
        }

    segment_features = torch.tensor(
        np.asarray(features_np[segment_start:segment_end], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    segment_masks = torch.tensor(
        np.asarray(masks_np[segment_start:segment_end, :SENSOR_DIM], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    all_metrics = MetricAccumulator()
    continuous_metrics = MetricAccumulator()
    binary_metrics = MetricAccumulator()
    prediction_buffer: list[torch.Tensor] = []
    target_buffer: list[torch.Tensor] = []
    mask_buffer: list[torch.Tensor] = []
    buffer_limit = 8192
    started_at = time.time()

    max_local_idx = segment_rows
    if max_steps_per_segment:
        max_local_idx = min(segment_rows, warmup_steps + max_steps_per_segment)

    with torch.no_grad(), autocast_context(device, amp):
        _, hidden = model.gru(segment_features[:warmup_steps].unsqueeze(0))
        for local_idx in range(warmup_steps, max_local_idx):
            prediction = model.head(hidden[-1]).squeeze(0).clamp(0.0, 1.0)
            target = segment_features[local_idx, :SENSOR_DIM]
            mask = segment_masks[local_idx]
            prediction_buffer.append(prediction.float())
            target_buffer.append(target.float())
            mask_buffer.append(mask.float())

            next_row = segment_features[local_idx : local_idx + 1].clone()
            next_row[:, :SENSOR_DIM] = prediction
            _, hidden = model.gru(next_row.unsqueeze(0), hidden)

            evaluated = local_idx - warmup_steps + 1
            if len(prediction_buffer) >= buffer_limit:
                flush_metric_buffers(
                    prediction_buffer,
                    target_buffer,
                    mask_buffer,
                    continuous_indices,
                    binary_indices,
                    near_zero_threshold,
                    relative_threshold,
                    all_metrics,
                    continuous_metrics,
                    binary_metrics,
                )
            if progress is not None:
                progress.update(1)
            if log_every_steps and evaluated % log_every_steps == 0:
                elapsed = time.time() - started_at
                print(
                    f"segment={segment_start}:{segment_end} step={evaluated}/{max_local_idx - warmup_steps} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    flush_metric_buffers(
        prediction_buffer,
        target_buffer,
        mask_buffer,
        continuous_indices,
        binary_indices,
        near_zero_threshold,
        relative_threshold,
        all_metrics,
        continuous_metrics,
        binary_metrics,
    )
    elapsed = time.time() - started_at
    return {
        "start": segment_start,
        "end": segment_end,
        "rows": segment_rows,
        "warmup_steps": warmup_steps,
        "evaluated_steps": max_local_idx - warmup_steps,
        "truncated": bool(max_steps_per_segment and max_local_idx < segment_rows),
        "skipped": False,
        "elapsed_s": elapsed,
        "accuracy": all_metrics.accuracy,
        "correct_points": all_metrics.correct,
        "valid_points": all_metrics.valid,
        "continuous_accuracy": continuous_metrics.accuracy,
        "continuous_correct_points": continuous_metrics.correct,
        "continuous_valid_points": continuous_metrics.valid,
        "binary_accuracy": binary_metrics.accuracy,
        "binary_correct_points": binary_metrics.correct,
        "binary_valid_points": binary_metrics.valid,
        "mae": all_metrics.mae,
        "continuous_mae": continuous_metrics.mae,
        "binary_mae": binary_metrics.mae,
    }


def aggregate_results(results: list[dict]) -> dict:
    all_metrics = MetricAccumulator()
    continuous_metrics = MetricAccumulator()
    binary_metrics = MetricAccumulator()
    evaluated_steps = 0
    elapsed_s = 0.0
    for item in results:
        if item.get("skipped"):
            continue
        evaluated_steps += int(item["evaluated_steps"])
        elapsed_s += float(item.get("elapsed_s", 0.0))
        all_metrics.update(
            int(item["correct_points"]),
            int(item["valid_points"]),
            float(item["mae"]) * int(item["valid_points"]),
        )
        continuous_metrics.update(
            int(item["continuous_correct_points"]),
            int(item["continuous_valid_points"]),
            float(item["continuous_mae"]) * int(item["continuous_valid_points"]),
        )
        binary_metrics.update(
            int(item["binary_correct_points"]),
            int(item["binary_valid_points"]),
            float(item["binary_mae"]) * int(item["binary_valid_points"]),
        )
    return {
        "evaluated_steps": evaluated_steps,
        "elapsed_s": elapsed_s,
        "accuracy": all_metrics.accuracy,
        "correct_points": all_metrics.correct,
        "valid_points": all_metrics.valid,
        "continuous_accuracy": continuous_metrics.accuracy,
        "continuous_correct_points": continuous_metrics.correct,
        "continuous_valid_points": continuous_metrics.valid,
        "binary_accuracy": binary_metrics.accuracy,
        "binary_correct_points": binary_metrics.correct,
        "binary_valid_points": binary_metrics.valid,
        "mae": all_metrics.mae,
        "continuous_mae": continuous_metrics.mae,
        "binary_mae": binary_metrics.mae,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    cache = args.cache or Path(config["cache"])
    warmup_steps = args.warmup_steps or int(config["warmup_steps"])
    max_gap_ms = args.max_gap_ms or int(config["max_gap_ms"])
    amp = args.amp or str(config.get("amp", "off"))

    split_dir = cache / args.split
    features_np = np.load(split_dir / "features.npy", mmap_mode="r")
    masks_np = np.load(split_dir / "input_mask.npy", mmap_mode="r")
    ts_ms = np.load(split_dir / "ts_ms.npy", mmap_mode="r")
    day_ranges = load_day_ranges(cache, args.split)
    requested_dates = set(args.dates) if args.dates else None

    model = GRUForecastModel(
        input_dim=int(features_np.shape[1]),
        output_dim=SENSOR_DIM,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    continuous_indices = id_to_indices(DEFAULT_CONTINUOUS_IDS).to(device)
    binary_indices = id_to_indices(DEFAULT_BINARY_IDS).to(device)

    selected_days = [
        day
        for day in day_ranges
        if int(day["rows"]) > 0 and (requested_dates is None or day["date"] in requested_dates)
    ]
    day_segments = build_eval_groups(
        selected_days=selected_days,
        ts_ms=ts_ms,
        max_gap_ms=max_gap_ms,
        rollout_mode=args.rollout_mode,
    )
    total_eval_steps = 0
    for _, segments in day_segments:
        for start, end in segments:
            segment_steps = max(0, end - start - warmup_steps)
            if args.max_steps_per_segment:
                segment_steps = min(segment_steps, args.max_steps_per_segment)
            total_eval_steps += segment_steps

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "cache": str(cache),
                "split": args.split,
                "dates": [day["date"] for day in selected_days],
                "warmup_steps": warmup_steps,
                "max_gap_ms": max_gap_ms,
                "rollout_mode": args.rollout_mode,
                "total_eval_steps": total_eval_steps,
                "max_steps_per_segment": args.max_steps_per_segment,
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )

    progress = None
    if not args.no_progress:
        progress = tqdm(total=total_eval_steps, desc="closed-loop", leave=True)
    day_results: list[dict] = []
    flat_segment_results: list[dict] = []
    try:
        for day, segments in day_segments:
            segment_results = []
            if args.quiet_segments:
                print(f"date={day} segments={len(segments)}", flush=True)
            for segment_start, segment_end in segments:
                if not args.quiet_segments:
                    print(f"date={day} segment={segment_start}:{segment_end} rows={segment_end - segment_start}", flush=True)
                segment_result = evaluate_segment(
                    model=model,
                    features_np=features_np,
                    masks_np=masks_np,
                    segment_start=segment_start,
                    segment_end=segment_end,
                    warmup_steps=warmup_steps,
                    device=device,
                    amp=amp,
                    continuous_indices=continuous_indices,
                    binary_indices=binary_indices,
                    near_zero_threshold=args.near_zero_threshold,
                    relative_threshold=args.relative_threshold,
                    log_every_steps=args.log_every_steps,
                    max_steps_per_segment=args.max_steps_per_segment,
                    progress=progress,
                )
                segment_results.append(segment_result)
                flat_segment_results.append(segment_result)
                if not args.quiet_segments and not segment_result.get("skipped"):
                    print(
                        f"date={day} segment={segment_start}:{segment_end} "
                        f"acc={segment_result['accuracy']:.6f} "
                        f"cont_acc={segment_result['continuous_accuracy']:.6f} "
                        f"bin_acc={segment_result['binary_accuracy']:.6f} "
                        f"mae={segment_result['mae']:.6f}",
                        flush=True,
                    )
            day_summary = aggregate_results(segment_results)
            day_results.append({"date": day, "segments": segment_results, **day_summary})
            print(
                f"date={day} day_acc={day_summary['accuracy']:.6f} "
                f"day_cont_acc={day_summary['continuous_accuracy']:.6f} "
                f"day_bin_acc={day_summary['binary_accuracy']:.6f} "
                f"day_mae={day_summary['mae']:.6f}",
                flush=True,
            )
    finally:
        if progress is not None:
            progress.close()

    overall = aggregate_results(flat_segment_results)
    result = {
        "checkpoint": str(args.checkpoint),
        "cache": str(cache),
        "split": args.split,
        "dates": [day["date"] for day in selected_days],
        "warmup_steps": warmup_steps,
        "max_gap_ms": max_gap_ms,
        "rollout_mode": args.rollout_mode,
        "max_steps_per_segment": args.max_steps_per_segment,
        "near_zero_threshold": args.near_zero_threshold,
        "relative_threshold": args.relative_threshold,
        "segment_reset_policy": reset_policy_text(args.rollout_mode, max_gap_ms),
        "overall": overall,
        "days": day_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.print_json:
        print(json.dumps(result, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {
                    "output_json": str(args.output_json),
                    "overall": overall,
                    "days": [
                        {
                            "date": item["date"],
                            "evaluated_steps": item["evaluated_steps"],
                            "accuracy": item["accuracy"],
                            "continuous_accuracy": item["continuous_accuracy"],
                            "binary_accuracy": item["binary_accuracy"],
                            "mae": item["mae"],
                            "continuous_mae": item["continuous_mae"],
                            "binary_mae": item["binary_mae"],
                        }
                        for item in day_results
                    ],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
