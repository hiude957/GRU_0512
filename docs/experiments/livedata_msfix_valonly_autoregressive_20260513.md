# livedata 毫秒修复与 5/8-5/11 全天自回归验证

记录日期：2026-05-13

## 背景

本次处理目标是修复 `livedata` 原始时间戳中毫秒位解析错误，并重新生成 `20260508` 至 `20260511` 的验证数据 cache，然后使用当前 best GRU checkpoint 跑全天自回归验证。

问题来源：

- `livedata` 原始时间格式存在 `2026/5/8 21:05:04:7`、`2026/5/9 11:30:25:87` 这类最后一段为 1 位或 2 位的毫秒字段。
- 旧解析方式把 `:7` 接成 `.7`，会被 Python `%f` 解释为 `700ms`。
- 实际业务含义应为毫秒字段左补零：`:7` 表示 `007ms`，`:87` 表示 `087ms`。

修复后解析结果：

| raw time | fixed parse |
|---|---|
| `2026/5/8 21:05:04:7` | `2026-05-08 21:05:04.007` |
| `2026/5/9 11:30:25:87` | `2026-05-09 11:30:25.087` |
| `2026/5/9 00:00:16.893` | `2026-05-09 00:00:16.893` |

对应代码修改：

- `scripts/build_aligned_livedata_110ms.py`
- `parse_abs_time()` 对冒号分隔的毫秒字段执行 `zfill(3)`。

## 原始数据校验

修复后重新扫描 `20260508` 和 `20260509` 的 raw livedata：

| 项目 | 结果 |
|---|---:|
| livedata 文件数 | `5` |
| raw 行数 | `172415` |
| 1 位毫秒行数 | `1685` |
| 2 位毫秒行数 | `15693` |
| 3 位毫秒行数 | `155037` |
| 文件内时间倒序数 | `0` |

结论：修复后 raw livedata 文件内时间顺序正常，不再由毫秒解析造成倒序。

## aligned 重建

重新生成修复后的 aligned 数据：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/build_aligned_livedata_110ms.py \
  --dataset dataset \
  --output outputs/aligned_livedata_110ms_msfix
```

输出目录：

`outputs/aligned_livedata_110ms_msfix`

全量 manifest 概览：

| 项目 | 数值 |
|---|---:|
| 日期范围 | `20260318` 至 `20260511` |
| 日期数 | `45` |
| total output rows | `10141506` |
| rejected APC rows | `58106` |
| rejected livedata rows | `0` |

`20260508` 至 `20260511` 对齐结果：

| date | rows | hidden_actions | accepted_apc | rejected_apc | accepted_livedata | rejected_livedata | source_counts |
|---|---:|---:|---:|---:|---:|---:|---|
| `20260508` | `489265` | `29` | `193824` | `0` | `68871` | `0` | `livedata_grid:295477, apc_grid:193788` |
| `20260509` | `784752` | `0` | `162065` | `0` | `103545` | `0` | `livedata_grid:622096, apc_grid:162656` |
| `20260510` | `360552` | `4064` | `43196` | `0` | `48570` | `0` | `livedata_grid:355452, apc_grid:5100` |
| `20260511` | `501443` | `0` | `27958` | `0` | `72599` | `0` | `livedata_grid:473471, apc_grid:27972` |

aligned `ts_ms` 顺序校验：

| date | rows | negative ts | duplicate ts | first ts_ms | last ts_ms |
|---|---:|---:|---:|---:|---:|
| `20260508` | `489265` | `0` | `0` | `1778230863200` | `1778284799830` |
| `20260509` | `784752` | `0` | `0` | `1778284800330` | `1778371199720` |
| `20260510` | `360552` | `0` | `0` | `1778371200220` | `1778410863030` |
| `20260511` | `501443` | `0` | `0` | `1778488607020` | `1778543999390` |

## val-only cache

因为此次时间解析问题只影响 `20260508` 至 `20260511` 的验证侧数据，训练 cache 不需要重建。中途停止了全量 cache 构建，改为只构建验证日期 cache。

先从新 aligned manifest 中过滤出 `20260508` 至 `20260511`：

`outputs/aligned_livedata_110ms_msfix/manifest_val_20260508_20260511.json`

filtered manifest 概览：

| 项目 | 数值 |
|---|---:|
| dates | `20260508, 20260509, 20260510, 20260511` |
| total rows | `2136012` |
| livedata_grid rows | `1746496` |
| apc_grid rows | `389516` |

构建 val-only cache：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/build_gru_cache.py \
  --aligned-dir outputs/aligned_livedata_110ms_msfix/aligned \
  --aligned-manifest outputs/aligned_livedata_110ms_msfix/manifest_val_20260508_20260511.json \
  --output outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly \
  --val-start 20260508 \
  --val-end 20260511
```

输出目录：

`outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly`

cache 校验：

| 项目 | 结果 |
|---|---:|
| train rows | `0` |
| val rows | `2136012` |
| val valid_targets | `2135991` |
| feature_dim | `401` |
| features shape | `(2136012, 401)` |
| masks_in_features | `False` |
| val ts negative | `0` |
| val ts duplicate | `0` |

说明：

- 这个 cache 只用于验证与自回归评估。
- 模型输入不包含 `mask_*` 位。
- `input_mask` 仍保留在 cache 中，用于有效 sensor 点统计和 loss/accuracy mask。

## 自回归评估设置

使用 checkpoint：

`runs/gru_0430_replicate_nomask/checkpoints/best.pt`

评估脚本：

`scripts/eval_gru_replicate_closed_loop_days.py`

本次为满足全天自回归要求，对评估脚本增加了 `--rollout-mode`：

| mode | 含义 |
|---|---|
| `segments` | 旧逻辑，按日期和 `ts_ms` gap 大于 `max_gap_ms` 的 segment 重置 |
| `day` | 每天一条自回归链，日内 segment/gap 不重置 |
| `continuous` | 所选日期合并为一条自回归链，跨天也不重置 |

共同参数：

| 参数 | 值 |
|---|---:|
| `warmup_steps` | `256` |
| `max_gap_ms` | `2000` |
| `near_zero_threshold` | `0.005` |
| `relative_threshold` | `0.20` |
| device | `cuda` |

accuracy 规则：

- 当 `abs(true) <= 0.005` 时，`abs(pred) < 0.005` 判为正确。
- 当 `abs(true) > 0.005` 时，`abs(pred - true) / abs(true) < 0.20` 判为正确。
- 只统计 `input_mask > 0` 的有效 sensor 点。

## 每天单独全天自回归

运行命令：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/eval_gru_replicate_closed_loop_days.py \
  --checkpoint runs/gru_0430_replicate_nomask/checkpoints/best.pt \
  --cache outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly \
  --split val \
  --dates 20260508 20260509 20260510 20260511 \
  --rollout-mode day \
  --quiet-segments \
  --log-every-steps 100000 \
  --output-json outputs/eval/gru_0430_replicate_nomask/msfix_valonly_day_chain_20260508_20260511.json \
  --no-progress
```

策略说明：

- 每天开头使用真实数据 warmup `256` 步。
- 每天内部只有一个自回归链。
- 日内 livedata segment 边界不重置。
- 日内长 gap 不重置。
- 跨天会重置，因为每天单独评估。

结果：

| date | evaluated_steps | accuracy | continuous_accuracy | binary_accuracy | mae | continuous_mae | binary_mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260508` | `489009` | `0.916982` | `0.888115` | `0.994925` | `0.013216` | `0.016805` | `0.003527` |
| `20260509` | `784496` | `0.927735` | `0.902173` | `0.996751` | `0.008941` | `0.011387` | `0.002338` |
| `20260510` | `360296` | `0.951972` | `0.934285` | `0.999728` | `0.003853` | `0.005209` | `0.000194` |
| `20260511` | `501187` | `0.935710` | `0.912253` | `0.999044` | `0.009882` | `0.013298` | `0.000658` |
| overall | `2134988` | `0.931235` | `0.906739` | `0.997373` | `0.009283` | `0.012034` | `0.001854` |

输出：

`outputs/eval/gru_0430_replicate_nomask/msfix_valonly_day_chain_20260508_20260511.json`

## 四天连续全天自回归

运行命令：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/eval_gru_replicate_closed_loop_days.py \
  --checkpoint runs/gru_0430_replicate_nomask/checkpoints/best.pt \
  --cache outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly \
  --split val \
  --dates 20260508 20260509 20260510 20260511 \
  --rollout-mode continuous \
  --quiet-segments \
  --log-every-steps 100000 \
  --output-json outputs/eval/gru_0430_replicate_nomask/msfix_valonly_continuous_chain_20260508_20260511.json \
  --no-progress
```

策略说明：

- 只在 `20260508` 开头使用真实数据 warmup `256` 步。
- `20260508` 至 `20260511` 合并为一条连续自回归链。
- 日内 segment 边界不重置。
- 跨天边界不重置。
- 5/9、5/10、5/11 的开头不再重新用真实 256 步 warmup，而是继承上一天末尾的预测 sensor 状态和 GRU hidden。

结果：

| range | evaluated_steps | accuracy | continuous_accuracy | binary_accuracy | mae | continuous_mae | binary_mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260508-20260511` | `2135756` | `0.924081` | `0.898285` | `0.993730` | `0.011290` | `0.013644` | `0.004935` |

输出：

`outputs/eval/gru_0430_replicate_nomask/msfix_valonly_continuous_chain_20260508_20260511.json`

## 对比与结论

| 评估方式 | reset 策略 | accuracy | continuous_accuracy | binary_accuracy | mae |
|---|---|---:|---:|---:|---:|
| 每天单独全天自回归 | 每天开头重置并 warmup，日内不重置 | `0.931235` | `0.906739` | `0.997373` | `0.009283` |
| 四天连续全天自回归 | 5/8 开头 warmup 一次，之后跨天不重置 | `0.924081` | `0.898285` | `0.993730` | `0.011290` |

结论：

1. 毫秒解析修复后，`20260508` 至 `20260511` 的 raw livedata、aligned TSV、val cache 时间顺序均无倒序和重复。
2. 只重建验证侧 cache 是合理的；本次问题不需要重建旧训练 cache。
3. 每天单独全天自回归的 overall accuracy 为 `0.931235`。
4. 四天连续自回归的 accuracy 为 `0.924081`，低于每天单独评估，说明跨天完全继承会带来额外误差积累。
5. continuous sensor 是主要误差来源；binary sensor 在两种评估方式下仍保持较高准确率。
6. 后续如果目标是生产式连续预测，需要重点观察连续 sensor 的长链漂移，而不仅是短窗口或单日 warmup 后的指标。

## 相关产物

代码：

- `scripts/build_aligned_livedata_110ms.py`
- `scripts/eval_gru_replicate_closed_loop_days.py`

数据与评估输出：

- `outputs/aligned_livedata_110ms_msfix`
- `outputs/aligned_livedata_110ms_msfix/manifest_val_20260508_20260511.json`
- `outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly`
- `outputs/eval/gru_0430_replicate_nomask/msfix_valonly_day_chain_20260508_20260511.json`
- `outputs/eval/gru_0430_replicate_nomask/msfix_valonly_continuous_chain_20260508_20260511.json`

注意：

- `outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix` 是中途停止的全量 cache 构建目录，不作为本次有效产物。
- 当前有效 cache 是 `outputs/cache/gru_110ms_nomask_val_20260508_20260511_msfix_valonly`。
