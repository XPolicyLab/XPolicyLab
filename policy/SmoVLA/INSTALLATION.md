# SmoVLA In XPolicyLab
``` bash
这份 policy 依赖上游 LeRobot v0.4.4。仅执行 `pip install -e .` 往往不够，常见缺失点有两类：

- LeRobot 的 SmolVLA 额外依赖没有安装。
- `av` / `ffmpeg` 相关 wheel 不可用时，系统里缺少编译依赖。

下面这套步骤是当前仓库下更完整、可复现的 Linux 安装方式。

## 1. Python 环境

建议使用 Python 3.10。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2. 系统依赖

LeRobot 官方安装文档里明确提到需要 ffmpeg；如果 `av` 需要本地编译，还需要 ffmpeg 开发头文件和基础编译工具。

```bash
sudo apt-get update
sudo apt-get install -y \
	git ffmpeg cmake build-essential pkg-config python3-dev \
	libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
	libswscale-dev libswresample-dev libavfilter-dev
```

说明：

- `ffmpeg` 最好带 `libsvtav1` 编码器；可用 `ffmpeg -encoders | grep svtav1` 检查。
- 如果你的机器上 `av` 能直接装 wheel，这些 dev 包不一定都会用到；但缺它们时最容易在安装阶段失败。

## 3. 安装 LeRobot 源码和 SmolVLA extra

只装基础包会漏掉 SmolVLA 所需的 `transformers`、`num2words`、`safetensors` 等 extra。这里要显式安装 `smolvla` extra。

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/SmoVLA
git clone -b v0.4.4 https://github.com/huggingface/lerobot.git smovla
cd smovla
pip install -e ".[smolvla]"
```

如果你还需要 LeRobot 的更多功能，可以改成：

```bash
pip install -e ".[smolvla,peft]"
```

不建议默认装 `.[all]`，它会拉起大量仿真、硬件和可选编译依赖。

## 4. 安装 XPolicyLab 侧依赖

回到当前工作区后，再安装 XPolicyLab 这一层。

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/SmoVLA/smovla
pip install ../../..
pip install h5py
```

如果你就是在 `/mnt/nfs/niantian/robodojo_test/XPolicyLab` 仓库里工作，也可以直接在仓库根目录执行：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install .
pip install h5py
```

## 5. 快速自检

```bash
python -c "import lerobot; print('lerobot ok')"
python -c "from lerobot.policies.factory import get_policy_class; print(get_policy_class('smolvla'))"
python -c "import av, transformers, safetensors, h5py; print('deps ok')"
```

## 常见问题

### 1) `pip install -e .` 成功，但运行时找不到 SmolVLA 相关模块

原因通常是你只安装了 LeRobot 基础包，没有安装 `.[smolvla]` extra。

### 2) `av` / ffmpeg 相关编译失败

优先检查：

- 系统里是否已安装 `ffmpeg`
- 上面的 `libav*` 开发包是否齐全
- `python3-dev`、`cmake`、`build-essential` 是否已安装

### 3) WSL 或精简系统里 `pynput` / 输入设备相关报错

这是 LeRobot 通用依赖问题；如果只做离线推理通常不是阻塞项，但完整环境建议参考上游 `smovla/docs/source/installation.mdx`。
```