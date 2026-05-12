# 项目工程架构

这一页描述当前仓库的代码、脚本、配置、输出、模型和实验记录组织方式。回答“代码放哪里”“脚本怎么分”“模型和结果放哪里”“调参记录怎么写”时优先读取这一页。

## 1. 总体原则

当前仓库采用 `src` 包 + `scripts` 入口的结构：

- `src/naura_gru/` 放可复用核心逻辑
- `scripts/` 放命令行入口，只做参数解析和调用核心逻辑
- `configs/` 放可提交的 YAML 配置
- `docs/experiments/` 放每次调参和训练结果的 Markdown 记录
- `outputs/`、`runs/`、`models/`、`cache/` 放大体积产物，并由 `.gitignore` 忽略

不要把数据文件、对齐结果、cache、checkpoint、预测结果或训练日志提交到 GitHub。

## 2. 代码目录职责

推荐包名是 `naura_gru`。

`src/naura_gru/data/`：

- `raw_readers.py`：读取 raw `log`、`apc`、`livedata`，恢复 APC 绝对时间
- `grid_align.py`：实现 `livedata` 覆盖范围内固定 `110ms` 网格、sensor 投影、全量 action 聚合、隐藏 action 状态更新、`evt_*` / `state_*`
- `defaults.py`：读取 `action_default.xlsx` 和 sensor 默认值；处理 `20260318` 首条 `livedata` 前的状态初始化
- `cache_builder.py`：把 aligned table 转成训练 cache arrays

`src/naura_gru/models/`：

- `gru.py`：当前主模型 GRU
- 后续新增模型时，每个模型单独文件，不把训练逻辑写进模型文件

`src/naura_gru/training/`：

- `dataset.py`：训练 Dataset、窗口采样、batch 组织
- `losses.py`：masked loss、sample weight、有效 target 处理
- `trainer.py`：训练循环、checkpoint、metrics 输出

`src/naura_gru/evaluation/`：

- `metrics.py`：归一化 sensor 准确率规则
- `closed_loop.py`：一个月无真实 sensor 修正的闭环滚动评估

`src/naura_gru/utils/`：

- `config.py`：加载 YAML 配置
- `logging.py`：日志和运行目录初始化
- `seed.py`：随机种子固定

## 3. 脚本职责

`scripts/01_build_aligned.py`：

- 输入 `dataset/`
- 输出 `outputs/aligned/`
- 负责调用 `grid_align.py`

`scripts/02_build_cache.py`：

- 输入 `outputs/aligned/`
- 输出 `outputs/cache/`
- 负责调用 `cache_builder.py`

`scripts/03_train_gru.py`：

- 输入 `outputs/cache/`
- 输出 `runs/<run_id>/`
- 保存 `config.yaml`、`metrics.json`、`train.log`、checkpoint

`scripts/04_eval_closed_loop.py`：

- 输入 `runs/<run_id>/` 或 checkpoint
- 输出 `outputs/eval/<run_id>/`
- 重点验证闭环滚动稳定性和准确率

`scripts/05_new_experiment_note.py`：

- 从 `runs/<run_id>/config.yaml` 和 `metrics.json` 生成 Markdown 草稿
- 输出 `docs/experiments/<run_id>.md`

## 4. 配置和输出约定

`configs/align_110ms.yaml`：数据对齐配置。

`configs/cache_dataset.yaml`：cache 构建配置。

`configs/train_gru_a100.yaml`：A100 80G 上的 GRU 起始训练配置。

`configs/eval_closed_loop.yaml`：闭环评估配置。

输出目录约定：

- aligned table：`outputs/aligned/<date>/`
- cache arrays：`outputs/cache/<cache_name>/`
- training run：`runs/<run_id>/`
- exported model：`models/exported/<model_name>/`
- evaluation result：`outputs/eval/<run_id>/`

`run_id` 推荐格式：

`YYYYMMDD_HHMMSS_short-name`

## 5. 调参 Markdown 记录

每次调参或训练保留一篇 Markdown：

`docs/experiments/<run_id>.md`

每篇记录包含：

- 目标：本次想验证什么
- 数据：日期范围、cache 版本、`livedata` 覆盖窗口规则、是否处理覆盖范围外隐藏 action、是否包含无 `livedata` 日期的状态继承
- 配置：模型参数、训练参数、关键数据处理参数
- 命令：实际运行命令
- 结果：loss、accuracy、closed-loop 指标、显存、耗时
- 结论：本次是否有效、问题是什么
- 下一步：下一次要改什么

`docs/experiments/README.md` 做总索引，至少记录：

- `run_id`
- 日期
- 主要改动
- best val loss
- accuracy
- closed-loop 结论
- 备注

## 6. uv 与 Git 约束

执行 Python 命令时使用 `uv`。

依赖相关命令必须设置：

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp
```

不要修改 uv 下载源。

GitHub 只提交：

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `.agents/`
- `.gitignore`
- `pyproject.toml`
- `uv.lock`

不要提交：

- `dataset/`
- `outputs/`
- `runs/`
- `models/`
- `cache/`
- checkpoint、日志、预测文件、`.npy`、`.tsv`、`.parquet`
