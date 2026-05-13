#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SENSOR_DIM = 150
DEFAULT_CONTINUOUS_IDS = (
    list(range(1, 22))
    + list(range(23, 93))
    + list(range(129, 146))
    + [149, 150]
)
DEFAULT_BINARY_IDS = [22] + list(range(93, 129)) + [146, 147, 148]


@dataclass(frozen=True)
class SplitArrays:
    features: np.ndarray
    input_mask: np.ndarray
    ts_ms: np.ndarray


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    mae: float
    continuous_mae: float
    binary_mae: float


class CurrentCacheRolloutDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        warmup_steps: int,
        rollout_steps: int,
        stride: int,
        max_gap_ms: int,
    ) -> None:
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        split_dir = cache_dir / split
        self.split = split
        self.warmup_steps = warmup_steps
        self.rollout_steps = rollout_steps
        self.total_window = warmup_steps + rollout_steps
        self.arrays = SplitArrays(
            features=np.load(split_dir / "features.npy", mmap_mode="r"),
            input_mask=np.load(split_dir / "input_mask.npy", mmap_mode="r"),
            ts_ms=np.load(split_dir / "ts_ms.npy", mmap_mode="r"),
        )
        self.feature_dim = int(self.arrays.features.shape[1])
        self.sensor_dim = SENSOR_DIM
        self.starts = self._build_starts(stride=stride, max_gap_ms=max_gap_ms)
        if len(self.starts) == 0:
            raise ValueError(f"No trainable windows found for split={split}")

    def _build_starts(self, stride: int, max_gap_ms: int) -> np.ndarray:
        ts_ms = self.arrays.ts_ms
        if len(ts_ms) < self.total_window:
            return np.empty((0,), dtype=np.int64)
        breaks = np.flatnonzero(np.diff(ts_ms) > max_gap_ms) + 1
        segment_starts = np.concatenate(([0], breaks)).astype(np.int64)
        segment_ends = np.concatenate((breaks, [len(ts_ms)])).astype(np.int64)
        starts: list[np.ndarray] = []
        for seg_start, seg_end in zip(segment_starts, segment_ends):
            max_start = int(seg_end - self.total_window)
            if max_start < seg_start:
                continue
            candidate = np.arange(seg_start, max_start + 1, stride, dtype=np.int64)
            if len(candidate):
                starts.append(candidate)
        if not starts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(starts)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start = int(self.starts[index])
        stop = start + self.total_window
        target_start = start + self.warmup_steps
        target_stop = target_start + self.rollout_steps
        return {
            "window_features": np.asarray(self.arrays.features[start:stop], dtype=np.float32).copy(),
            "target_sensor": np.asarray(
                self.arrays.features[target_start:target_stop, :SENSOR_DIM],
                dtype=np.float32,
            ).copy(),
            "target_mask": np.asarray(
                self.arrays.input_mask[target_start:target_stop, :SENSOR_DIM],
                dtype=np.float32,
            ).copy(),
        }


class GRUForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the GRU_0430 absolute active/change rollout training on the GRU_0512 cache."
    )
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache/gru_110ms_nomask_val_20260508_20260511"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gru_0430_replicate_nomask"))
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--val-stride", type=int, default=64)
    parser.add_argument("--max-gap-ms", type=int, default=2000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default="bf16")
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument("--log-every-steps", type=int, default=0)
    parser.add_argument("--continuous-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--active-continuous-weight", type=float, default=2.0)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    parser.add_argument("--change-loss-weight", type=float, default=3.0)
    parser.add_argument("--change-threshold", type=float, default=0.005)
    parser.add_argument("--max-loss-weight", type=float, default=5.0)
    parser.add_argument("--early-stopping-metric", choices=("val_loss", "val_mae"), default="val_mae")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def id_to_indices(sensor_ids: list[int] | tuple[int, ...]) -> torch.Tensor:
    return torch.tensor([sensor_id - 1 for sensor_id in sensor_ids], dtype=torch.long)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    previous_target: torch.Tensor,
    continuous_indices: torch.Tensor,
    binary_indices: torch.Tensor,
    continuous_loss_weight: float,
    binary_loss_weight: float,
    active_continuous_weight: float,
    active_threshold: float,
    change_loss_weight: float,
    change_threshold: float,
    max_loss_weight: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = torch.zeros_like(mask)
    weights.index_fill_(2, continuous_indices, continuous_loss_weight)
    weights.index_fill_(2, binary_indices, binary_loss_weight)
    continuous_target = target.index_select(2, continuous_indices)
    continuous_previous = previous_target.index_select(2, continuous_indices)
    continuous_weights = weights.index_select(2, continuous_indices)
    if active_continuous_weight != 1.0:
        continuous_weights = torch.where(
            continuous_target > active_threshold,
            continuous_weights * active_continuous_weight,
            continuous_weights,
        )
    if change_loss_weight != 1.0:
        continuous_change = (continuous_target - continuous_previous).abs()
        continuous_weights = torch.where(
            continuous_change > change_threshold,
            continuous_weights * change_loss_weight,
            continuous_weights,
        )
    weights[:, :, continuous_indices] = continuous_weights
    if max_loss_weight > 0:
        weights = weights.clamp(max=max_loss_weight)
    weighted_mask = mask * weights
    return (loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def autoregressive_predictions(
    model: GRUForecastModel,
    window_features: torch.Tensor,
    warmup_steps: int,
    rollout_steps: int,
    sensor_dim: int,
) -> torch.Tensor:
    _, hidden = model.gru(window_features[:, :warmup_steps, :])
    predictions: list[torch.Tensor] = []
    for step in range(rollout_steps):
        prediction = model.head(hidden[-1]).clamp(0.0, 1.0)
        predictions.append(prediction)
        if step == rollout_steps - 1:
            continue
        next_row = window_features[:, warmup_steps + step, :].clone()
        next_row[:, :sensor_dim] = prediction
        _, hidden = model.gru(next_row.unsqueeze(1), hidden)
    return torch.stack(predictions, dim=1)


def rollout_loss(
    model: GRUForecastModel,
    window_features: torch.Tensor,
    target_sensor: torch.Tensor,
    target_mask: torch.Tensor,
    warmup_steps: int,
    rollout_steps: int,
    sensor_dim: int,
    continuous_indices: torch.Tensor,
    binary_indices: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = autoregressive_predictions(
        model=model,
        window_features=window_features,
        warmup_steps=warmup_steps,
        rollout_steps=rollout_steps,
        sensor_dim=sensor_dim,
    )
    previous_sensor = torch.cat(
        (
            window_features[:, warmup_steps - 1 : warmup_steps, :sensor_dim],
            target_sensor[:, :-1, :sensor_dim],
        ),
        dim=1,
    )
    loss = masked_weighted_huber(
        prediction=predictions,
        target=target_sensor,
        mask=target_mask,
        previous_target=previous_sensor,
        continuous_indices=continuous_indices,
        binary_indices=binary_indices,
        continuous_loss_weight=args.continuous_loss_weight,
        binary_loss_weight=args.binary_loss_weight,
        active_continuous_weight=args.active_continuous_weight,
        active_threshold=args.active_threshold,
        change_loss_weight=args.change_loss_weight,
        change_threshold=args.change_threshold,
        max_loss_weight=args.max_loss_weight,
    )
    mae = masked_mae(predictions, target_sensor, target_mask)
    continuous_mae = masked_mae(
        predictions.index_select(2, continuous_indices),
        target_sensor.index_select(2, continuous_indices),
        target_mask.index_select(2, continuous_indices),
    )
    binary_mae = masked_mae(
        predictions.index_select(2, binary_indices),
        target_sensor.index_select(2, binary_indices),
        target_mask.index_select(2, binary_indices),
    )
    return loss, mae, continuous_mae, binary_mae


def evaluate_loader(
    model: GRUForecastModel,
    loader: DataLoader,
    device: torch.device,
    continuous_indices: torch.Tensor,
    binary_indices: torch.Tensor,
    args: argparse.Namespace,
) -> EvalMetrics:
    model.eval()
    total_loss = total_mae = total_continuous_mae = total_binary_mae = 0.0
    steps = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            window_features = batch["window_features"].to(device, non_blocking=True)
            target_sensor = batch["target_sensor"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            loss, mae, continuous_mae, binary_mae = rollout_loss(
                model=model,
                window_features=window_features,
                target_sensor=target_sensor,
                target_mask=target_mask,
                warmup_steps=args.warmup_steps,
                rollout_steps=args.rollout_steps,
                sensor_dim=SENSOR_DIM,
                continuous_indices=continuous_indices,
                binary_indices=binary_indices,
                args=args,
            )
            total_loss += float(loss.item())
            total_mae += float(mae.item())
            total_continuous_mae += float(continuous_mae.item())
            total_binary_mae += float(binary_mae.item())
            steps += 1
            if args.max_val_batches and step >= args.max_val_batches:
                break
    if steps == 0:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0)
    return EvalMetrics(
        loss=total_loss / steps,
        mae=total_mae / steps,
        continuous_mae=total_continuous_mae / steps,
        binary_mae=total_binary_mae / steps,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CurrentCacheRolloutDataset(
        args.cache,
        "train",
        warmup_steps=args.warmup_steps,
        rollout_steps=args.rollout_steps,
        stride=args.train_stride,
        max_gap_ms=args.max_gap_ms,
    )
    val_dataset = CurrentCacheRolloutDataset(
        args.cache,
        "val",
        warmup_steps=args.warmup_steps,
        rollout_steps=args.rollout_steps,
        stride=args.val_stride,
        max_gap_ms=args.max_gap_ms,
    )
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    model = GRUForecastModel(
        input_dim=train_dataset.feature_dim,
        output_dim=SENSOR_DIM,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    continuous_indices = id_to_indices(DEFAULT_CONTINUOUS_IDS).to(device)
    binary_indices = id_to_indices(DEFAULT_BINARY_IDS).to(device)

    config = {
        "source_run": "/vepfs-mlp2/mlp-public/250259/lyh/GRU_0430/outputs_all_sensors_absolute_active_change_w2_w3_scratch",
        "cache": str(args.cache),
        "feature_dim": train_dataset.feature_dim,
        "sensor_dim": SENSOR_DIM,
        "continuous_sensor_ids": DEFAULT_CONTINUOUS_IDS,
        "binary_sensor_ids": DEFAULT_BINARY_IDS,
        "continuous_dim": len(DEFAULT_CONTINUOUS_IDS),
        "binary_dim": len(DEFAULT_BINARY_IDS),
        "warmup_steps": args.warmup_steps,
        "rollout_steps": args.rollout_steps,
        "train_stride": args.train_stride,
        "val_stride": args.val_stride,
        "max_gap_ms": args.max_gap_ms,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "target_mode": "absolute_sensor",
        "continuous_loss_weight": args.continuous_loss_weight,
        "binary_loss_weight": args.binary_loss_weight,
        "active_continuous_weight": args.active_continuous_weight,
        "active_threshold": args.active_threshold,
        "change_loss_weight": args.change_loss_weight,
        "change_threshold": args.change_threshold,
        "max_loss_weight": args.max_loss_weight,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "max_val_batches": args.max_val_batches,
        "log_every_steps": args.log_every_steps,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "amp": args.amp,
        "seed": args.seed,
        "note": "Replicates GRU_0430 absolute active/change rollout training on GRU_0512 cache; mask columns are not model inputs.",
    }
    (checkpoint_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "device": str(device),
                "feature_dim": train_dataset.feature_dim,
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "batch_size": args.batch_size,
                "warmup_steps": args.warmup_steps,
                "rollout_steps": args.rollout_steps,
                "target_mode": "absolute_sensor",
            },
            indent=2,
        )
    )

    history: list[dict] = []
    best_score = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_mae = total_continuous_mae = total_binary_mae = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, disable=args.no_progress)
        for step, batch in enumerate(progress, start=1):
            window_features = batch["window_features"].to(device, non_blocking=True)
            target_sensor = batch["target_sensor"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                loss, mae, continuous_mae, binary_mae = rollout_loss(
                    model=model,
                    window_features=window_features,
                    target_sensor=target_sensor,
                    target_mask=target_mask,
                    warmup_steps=args.warmup_steps,
                    rollout_steps=args.rollout_steps,
                    sensor_dim=SENSOR_DIM,
                    continuous_indices=continuous_indices,
                    binary_indices=binary_indices,
                    args=args,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item())
            total_mae += float(mae.item())
            total_continuous_mae += float(continuous_mae.item())
            total_binary_mae += float(binary_mae.item())
            steps += 1
            if not args.no_progress:
                progress.set_postfix(loss=f"{loss.item():.5f}", mae=f"{mae.item():.5f}")
            if args.log_every_steps and step % args.log_every_steps == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={loss.item():.6f} mae={mae.item():.6f} "
                    f"cont_mae={continuous_mae.item():.6f} bin_mae={binary_mae.item():.6f}",
                    flush=True,
                )
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break

        train_loss = total_loss / max(steps, 1)
        train_mae = total_mae / max(steps, 1)
        train_continuous_mae = total_continuous_mae / max(steps, 1)
        train_binary_mae = total_binary_mae / max(steps, 1)
        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            continuous_indices=continuous_indices,
            binary_indices=binary_indices,
            args=args,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_continuous_mae": train_continuous_mae,
            "train_binary_mae": train_binary_mae,
            "val_loss": val_metrics.loss,
            "val_mae": val_metrics.mae,
            "val_continuous_mae": val_metrics.continuous_mae,
            "val_binary_mae": val_metrics.binary_mae,
        }
        history.append(record)
        (checkpoint_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_mae={train_mae:.6f} "
            f"train_cont_mae={train_continuous_mae:.6f} train_bin_mae={train_binary_mae:.6f} "
            f"val_loss={val_metrics.loss:.6f} val_mae={val_metrics.mae:.6f} "
            f"val_cont_mae={val_metrics.continuous_mae:.6f} val_bin_mae={val_metrics.binary_mae:.6f}",
            flush=True,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        current_score = float(record[args.early_stopping_metric])
        if current_score < best_score - args.min_delta:
            best_score = current_score
            stale_epochs = 0
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1
        if args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
            print(
                f"early_stopping epoch={epoch} metric={args.early_stopping_metric} "
                f"best={best_score:.6f} patience={args.early_stopping_patience}",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
