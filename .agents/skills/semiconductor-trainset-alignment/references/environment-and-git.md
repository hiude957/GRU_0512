# 环境、uv 与 Git 管理规则

这一页描述当前项目的开发环境、依赖管理约束和 GitHub 上传规则。回答环境配置、依赖安装、缓存目录、Git 提交或 GitHub 上传问题时优先读取这一页。

## 1. 开发环境

当前项目运行在火山引擎开发机上。

主要硬件：

- GPU：NVIDIA A100 80G
- CPU：Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz

这些硬件信息用于推荐训练参数和并行加载参数，不应被写成所有部署环境都必须一致。

## 2. uv 依赖管理

项目使用 `uv` 管理 Python 依赖。

规则：

- 安装、同步或运行依赖相关命令时使用 `uv`
- 不要修改 `uv` 下载源
- 不要修改镜像源、`index-url`、extra index 或相关源配置
- 默认认为下载源已经配置好

## 3. uv 缓存和临时目录

所有 `uv` 下载缓存使用：

`UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache`

所有可能下载、构建、解压依赖的临时文件使用：

`TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp`

执行依赖相关命令前，确保这两个目录存在。

示例命令形态：

```bash
mkdir -p /vepfs-mlp2/mlp-public/250259/lyh/uv-cache /vepfs-mlp2/mlp-public/250259/lyh/uv-tmp
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp uv sync
```

不要把 `uv-cache` 或 `uv-tmp` 放进 Git 管理。

## 4. GitHub 上传规则

GitHub 只上传代码类内容。

应提交：

- 源代码
- 配置文件
- 文档
- `.gitignore`
- `pyproject.toml`
- `uv.lock`
- 必要的小型示例或模板文件

不应提交：

- `dataset/`
- 原始数据文件
- 对齐后的训练数据
- cache 目录
- 模型 checkpoint
- 训练日志
- tensorboard / wandb 本地输出
- 临时文件
- 大体积运行产物

如果用户问是否上传 `uv.lock`，回答应为上传。`uv.lock` 用于保证训练环境可复现。

## 5. 当前仓库状态

当前仓库已配置 GitHub 远程仓库。

已知状态：

- remote：`origin`
- 当前分支：`main`
- 当前 `dataset/` 尚未被 Git 跟踪

因此只需要通过 `.gitignore` 忽略数据目录，不需要对 `dataset/` 执行 `git rm --cached`，除非之后发现数据已经被提交或被跟踪。
