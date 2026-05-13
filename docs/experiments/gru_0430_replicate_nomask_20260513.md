# GRU 0430 复刻训练记录 - no-mask cache

记录日期：2026-05-13

## 目标

本次实验目标是复刻旧目录中的较优训练方式：

`/vepfs-mlp2/mlp-public/250259/lyh/GRU_0430/outputs_all_sensors_absolute_active_change_w2_w3_scratch`

但数据改用当前已经重新对齐和缓存后的 `GRU_0512` 数据：

`outputs/cache/gru_110ms_nomask_val_20260508_20260511`

核心变化是：模型输入不再包含 `mask_*` 位，只使用真实输入特征；`target_mask` 仍用于 loss/MAE 的有效点筛选。

## 数据与窗口

- cache：`outputs/cache/gru_110ms_nomask_val_20260508_20260511`
- 验证集：按 cache 命名，为 `2026-05-08` 至 `2026-05-11`
- `feature_dim`：`401`
- `sensor_dim`：`150`
- 训练窗口数：`997036`
- 验证窗口数：`29633`
- 固定网格：`110ms`
- `warmup_steps`：`256`，约 `28.16s`
- `rollout_steps`：`64`，约 `7.04s`
- 单个窗口总长度：`320` steps，约 `35.20s`
- `train_stride`：`8`
- `val_stride`：`64`
- `max_gap_ms`：`2000`

窗口构建规则：

- 在 `ts_ms` 相邻差值大于 `2000ms` 的位置切分 segment。
- 滑动窗口只在同一个 segment 内生成，不跨越长 gap。
- warmup 段使用真实历史特征初始化 GRU hidden。
- rollout 段进行自回归预测，预测出的 sensor 回灌到下一步输入。
- 真实 sensor 只用于 rollout 段计算 loss/MAE。

sensor 分组：

- continuous sensor：`1-21, 23-92, 129-145, 149-150`，共 `110` 个。
- binary sensor：`22, 93-128, 146-148`，共 `40` 个。

## 模型参数

- 模型：`GRUForecastModel`
- 输入维度：`401`
- 输出维度：`150`
- `hidden_size`：`256`
- `num_layers`：`2`
- `dropout`：`0.1`
- 输出头：`LayerNorm(hidden_size) + Linear(hidden_size, 150)`
- 输出初始化：head 的 weight 和 bias 初始化为 `0`
- 预测值：rollout 时 clamp 到 `[0, 1]`

## 训练参数

- 训练脚本：`scripts/train_gru_replicate_0430.py`
- 输出目录：`runs/gru_0430_replicate_nomask`
- optimizer：`AdamW`
- `lr`：`3e-4`
- `weight_decay`：`1e-4`
- `batch_size`：`64`
- `epochs`：原计划 `13`
- 实际完成：`8` 个 epoch，随后手动停止
- `num_workers`：`4`
- mixed precision：`bf16`
- `gradient_clip`：`1.0`
- `seed`：`42`
- `max_val_batches`：`200`
- `log_every_steps`：`500`
- `early_stopping_metric`：`val_mae`
- `early_stopping_patience`：`0`，脚本本身不自动 early stop
- `save_every_epoch`：启用

实际运行命令：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/train_gru_replicate_0430.py \
  --output-dir runs/gru_0430_replicate_nomask \
  --epochs 13 \
  --batch-size 64 \
  --num-workers 4 \
  --amp bf16 \
  --save-every-epoch \
  --log-every-steps 500 \
  --no-progress
```

## Loss 与指标

训练目标为 `absolute_sensor`，即直接预测归一化后的 sensor 绝对值。

loss 使用 masked weighted Huber：

- 基础 loss：`smooth_l1_loss`
- 有效性筛选：`target_mask`
- continuous 基础权重：`1.0`
- binary 基础权重：`1.0`
- active continuous 加权：
  - 当 continuous target `> 0.01` 时乘以 `2.0`
- change-aware 加权：
  - 当 continuous target 与 previous target 的绝对变化 `> 0.005` 时乘以 `3.0`
- 最大 loss weight：`5.0`

记录指标：

- `train_loss`
- `train_mae`
- `train_continuous_mae`
- `train_binary_mae`
- `val_loss`
- `val_mae`
- `val_continuous_mae`
- `val_binary_mae`

其中 MAE 均按 `target_mask` 统计有效 sensor 点。

## 逐轮结果

| epoch | train_loss | train_mae | train_cont_mae | train_bin_mae | val_loss | val_mae | val_cont_mae | val_bin_mae | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.001086 | 0.008720 | 0.010221 | 0.004669 | 0.000939 | 0.008109 | 0.010755 | 0.000964 | 初始收敛 |
| 2 | 0.000490 | 0.005112 | 0.006362 | 0.001737 | 0.000914 | 0.007534 | 0.009973 | 0.000949 | 验证改善 |
| 3 | 0.000431 | 0.004537 | 0.005634 | 0.001574 | 0.000902 | 0.007251 | 0.009597 | 0.000919 | 验证改善 |
| 4 | 0.000400 | 0.004193 | 0.005188 | 0.001506 | 0.000902 | 0.006939 | 0.009181 | 0.000887 | 验证改善 |
| 5 | 0.000380 | 0.003980 | 0.004928 | 0.001422 | 0.000889 | 0.006704 | 0.008849 | 0.000913 | 验证改善 |
| 6 | 0.000368 | 0.003814 | 0.004703 | 0.001413 | 0.000844 | 0.006630 | 0.008758 | 0.000884 | 验证小幅改善 |
| 7 | 0.000359 | 0.003701 | 0.004552 | 0.001403 | 0.000820 | 0.006515 | 0.008605 | 0.000870 | best |
| 8 | 0.000357 | 0.003649 | 0.004483 | 0.001397 | 0.000858 | 0.006574 | 0.008654 | 0.000958 | 验证变差，停止 |

## 停止依据

训练集指标从 epoch 1 到 epoch 8 持续下降：

- `train_mae`：`0.008720 -> 0.003649`

但验证集在 epoch 7 达到最好后，epoch 8 反而变差：

- epoch 7：`val_mae = 0.0065145586`
- epoch 8：`val_mae = 0.0065744354`

因此判断当前训练已经进入边际收益很小甚至开始过拟合的阶段。epoch 8 完整写入后，手动中断后续训练，保留 epoch 7 对应的 `best.pt`。

## 产物

run 目录：

`runs/gru_0430_replicate_nomask`

checkpoint 目录：

`runs/gru_0430_replicate_nomask/checkpoints`

主要文件：

- `best.pt`：epoch 7，对应当前 best validation MAE
- `epoch_001.pt` 至 `epoch_008.pt`：逐轮 checkpoint
- `last.pt`：epoch 8
- `metrics.json`：逐轮指标
- `train_config.json`：本次训练配置

目录大小约 `36M`。

注意：

- `best.pt` 更新时间为 `2026-05-13 06:16`，对应 epoch 7。
- `last.pt` 更新时间为 `2026-05-13 06:47`，对应 epoch 8，但不是最佳模型。
- 手动中断发生在 epoch 8 保存完成后的下一轮训练中，不影响 epoch 1-8 的 checkpoint 和 metrics。

## 验证集 Accuracy

使用 epoch 7 的 `best.pt` 在完整验证集上重新跑自回归 accuracy：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/eval_gru_replicate_accuracy.py \
  --checkpoint runs/gru_0430_replicate_nomask/checkpoints/best.pt \
  --split val \
  --num-workers 4 \
  --no-progress \
  --output-json outputs/eval/gru_0430_replicate_nomask/val_accuracy_best.json
```

准确率规则：

- 当 `abs(true) > 0.005` 时，`abs(pred - true) / abs(true) < 0.20` 判为正确。
- 当 `abs(true) <= 0.005` 时，`abs(pred) < 0.005` 判为正确。
- 只统计 `target_mask > 0` 的有效 sensor 点。

结果：

| split | windows | batches | valid_points | correct_points | accuracy | continuous_accuracy | binary_accuracy | mae | continuous_mae | binary_mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 29633 | 464 | 280683776 | 266472010 | 0.949367 | 0.930879 | 0.999286 | 0.005037 | 0.006676 | 0.000613 |

对应输出：

`outputs/eval/gru_0430_replicate_nomask/val_accuracy_best.json`

## 资源监控

训练期间 GPU 状态稳定：

- GPU：`NVIDIA A100-SXM4-80GB`
- 显存：约 `22.9GB / 80GB`
- GPU 温度：约 `30C`
- GPU 利用率采样：约 `26%-32%`
- 训练时长：约 `4h10m`

没有观察到显存持续增长或进程卡死。

## 结论

本次 no-mask cache 上复刻旧训练策略是有效的。验证集 `val_mae` 从 epoch 1 的 `0.008109` 降到 epoch 7 的 `0.006515`，随后 epoch 8 回升，因此当前应采用 epoch 7 的 `best.pt`。

下一步不建议继续单纯增加 epoch。更有价值的方向是：

- 使用 `best.pt` 做 `2026-05-08` 至 `2026-05-11` 的闭环自回归验证。
- 观察 closed-loop 中误差是否随时间漂移。
- 如果二值 sensor 对验证波动影响较大，可尝试降低 `binary_loss_weight`，例如 `0.5` 或 `0.2`。
- 如果连续 sensor 的动作后变化仍预测不够好，可保留 change-aware loss，再单独调整 `active_continuous_weight` 和 `change_loss_weight`。
