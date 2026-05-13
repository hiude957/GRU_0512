#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_gru_replicate_0430 import (
    DEFAULT_BINARY_IDS,
    DEFAULT_CONTINUOUS_IDS,
    SENSOR_DIM,
    CurrentCacheRolloutDataset,
    GRUForecastModel,
    autocast_context,
    autoregressive_predictions,
    id_to_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GRU replicate checkpoint accuracy on a cache split."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/gru_0430_replicate_nomask/checkpoints/best.pt"),
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-gap-ms", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default=None)
    parser.add_argument("--near-zero-threshold", type=float, default=0.005)
    parser.add_argument("--relative-threshold", type=float, default=0.20)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval/gru_0430_replicate_nomask/val_accuracy_best.json"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    near_zero_threshold: float,
    relative_threshold: float,
) -> tuple[int, int]:
    valid = mask > 0
    if not bool(valid.any()):
        return 0, 0
    abs_target = target.abs()
    near_zero = abs_target <= near_zero_threshold
    correct_near_zero = prediction.abs() < near_zero_threshold
    relative_error = (prediction - target).abs() / abs_target.clamp_min(near_zero_threshold)
    correct_nonzero = relative_error < relative_threshold
    correct = torch.where(near_zero, correct_near_zero, correct_nonzero) & valid
    return int(correct.sum().item()), int(valid.sum().item())


def masked_abs_error_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, int]:
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return 0.0, 0
    return float(((prediction - target).abs() * mask).sum().item()), valid_count


def rate(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    cache = args.cache or Path(config["cache"])
    warmup_steps = args.warmup_steps or int(config["warmup_steps"])
    rollout_steps = args.rollout_steps or int(config["rollout_steps"])
    max_gap_ms = args.max_gap_ms or int(config["max_gap_ms"])
    batch_size = args.batch_size or int(config["batch_size"])
    if args.stride is not None:
        stride = args.stride
    else:
        stride = int(config["val_stride"] if args.split == "val" else config["train_stride"])
    amp = args.amp or str(config.get("amp", "off"))

    dataset = CurrentCacheRolloutDataset(
        cache_dir=cache,
        split=args.split,
        warmup_steps=warmup_steps,
        rollout_steps=rollout_steps,
        stride=stride,
        max_gap_ms=max_gap_ms,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = GRUForecastModel(
        input_dim=dataset.feature_dim,
        output_dim=SENSOR_DIM,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    continuous_indices = id_to_indices(DEFAULT_CONTINUOUS_IDS).to(device)
    binary_indices = id_to_indices(DEFAULT_BINARY_IDS).to(device)

    correct_all = total_all = 0
    correct_continuous = total_continuous = 0
    correct_binary = total_binary = 0
    abs_error_all = abs_error_continuous = abs_error_binary = 0.0
    mae_count_all = mae_count_continuous = mae_count_binary = 0
    batches = 0

    progress = tqdm(loader, desc=f"eval {args.split}", leave=False, disable=args.no_progress)
    with torch.no_grad():
        for batch_index, batch in enumerate(progress, start=1):
            window_features = batch["window_features"].to(device, non_blocking=True)
            target_sensor = batch["target_sensor"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            with autocast_context(device, amp):
                prediction = autoregressive_predictions(
                    model=model,
                    window_features=window_features,
                    warmup_steps=warmup_steps,
                    rollout_steps=rollout_steps,
                    sensor_dim=SENSOR_DIM,
                )

            batch_correct, batch_total = accuracy_counts(
                prediction=prediction,
                target=target_sensor,
                mask=target_mask,
                near_zero_threshold=args.near_zero_threshold,
                relative_threshold=args.relative_threshold,
            )
            correct_all += batch_correct
            total_all += batch_total
            err_sum, err_count = masked_abs_error_sum(prediction, target_sensor, target_mask)
            abs_error_all += err_sum
            mae_count_all += err_count

            pred_continuous = prediction.index_select(2, continuous_indices)
            target_continuous = target_sensor.index_select(2, continuous_indices)
            mask_continuous = target_mask.index_select(2, continuous_indices)
            batch_correct, batch_total = accuracy_counts(
                prediction=pred_continuous,
                target=target_continuous,
                mask=mask_continuous,
                near_zero_threshold=args.near_zero_threshold,
                relative_threshold=args.relative_threshold,
            )
            correct_continuous += batch_correct
            total_continuous += batch_total
            err_sum, err_count = masked_abs_error_sum(pred_continuous, target_continuous, mask_continuous)
            abs_error_continuous += err_sum
            mae_count_continuous += err_count

            pred_binary = prediction.index_select(2, binary_indices)
            target_binary = target_sensor.index_select(2, binary_indices)
            mask_binary = target_mask.index_select(2, binary_indices)
            batch_correct, batch_total = accuracy_counts(
                prediction=pred_binary,
                target=target_binary,
                mask=mask_binary,
                near_zero_threshold=args.near_zero_threshold,
                relative_threshold=args.relative_threshold,
            )
            correct_binary += batch_correct
            total_binary += batch_total
            err_sum, err_count = masked_abs_error_sum(pred_binary, target_binary, mask_binary)
            abs_error_binary += err_sum
            mae_count_binary += err_count

            batches += 1
            if args.max_batches and batch_index >= args.max_batches:
                break

    result = {
        "checkpoint": str(args.checkpoint),
        "cache": str(cache),
        "split": args.split,
        "windows": len(dataset),
        "batches": batches,
        "batch_size": batch_size,
        "stride": stride,
        "warmup_steps": warmup_steps,
        "rollout_steps": rollout_steps,
        "max_gap_ms": max_gap_ms,
        "near_zero_threshold": args.near_zero_threshold,
        "relative_threshold": args.relative_threshold,
        "accuracy": rate(correct_all, total_all),
        "correct_points": correct_all,
        "valid_points": total_all,
        "continuous_accuracy": rate(correct_continuous, total_continuous),
        "continuous_correct_points": correct_continuous,
        "continuous_valid_points": total_continuous,
        "binary_accuracy": rate(correct_binary, total_binary),
        "binary_correct_points": correct_binary,
        "binary_valid_points": total_binary,
        "mae": abs_error_all / max(mae_count_all, 1),
        "continuous_mae": abs_error_continuous / max(mae_count_continuous, 1),
        "binary_mae": abs_error_binary / max(mae_count_binary, 1),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
