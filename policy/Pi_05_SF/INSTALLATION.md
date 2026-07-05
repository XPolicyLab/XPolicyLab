# Pi_05_SF 安装配置

本文只说明当前 `Pi_05_SF` policy 包需要的环境和路径。训练、离线 cache 和 XPolicyLab 启动命令见 [README.md](README.md)。

## 1. 安装 OpenPI-SF 环境

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
cd "$XPL_ROOT/policy/Pi_05_SF/open_sf"

UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

如果只做 XPolicyLab 推理，也需要保证外层 `eval.sh` 启动时能找到：

- `open_sf/src`
- `open_sf/packages/openpi-client/src`
- `open_sf/src/vggt`

`eval.sh` 会自动把这些目录加入 `PYTHONPATH`。

## 2. 安装 XPolicyLab

在 XPolicyLab 仓库中安装其本体依赖：

```bash
cd "$XPL_ROOT"
uv pip install -e .
```

运行 `Pi_05_SF` policy server 时指定：

```bash
cd "$XPL_ROOT/policy/Pi_05_SF"
bash eval.sh
```
