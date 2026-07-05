# Being_H05


## 环境配置

推荐参考XPolicyLab/policy/Being_H05/Being-H/README.md配置环境



## 调试 / 启动步骤

当前 `XPolicyLab/debug_env_client.py` 是一个 **mock env** 调试入口，作用是验证：

- XPolicyLab policy server 能否正常启动
- `Being_H05/model.py` 能否正常加载 checkpoint
- obs/action 接口是否对齐

### 1. 启动 XPolicyLab policy server

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate beingh
cd /share/being-transfer/users/yiqing

python XPolicyLab/setup_policy_server.py \
  --config_path /share/being-transfer/users/yiqing/XPolicyLab/policy/Being_H05/deploy.yml \
  --overrides \
  port=6000 \
  policy_name=Being_H05 \
  task_name=adjust_bottle \
  env_cfg_type='aloha-agilex' \
  action_type='joint' \
  model_path='/share/being-transfer/users/yiqing/checkpoints/post-robotwin_clean_BH05-2B_chunk-16_20260403_191158/0150000' \
  data_config_name='robodojo_qpos' \
  bench_name='robodojo_posttrain' \
  embodiment_tag='new_embodiment' \
  prompt_template='long' \
  device='cuda'
```

### 2. 启动 XPolicyLab client

```bash
python XPolicyLab/debug_env_client.py \
  --task_name adjust_bottle \
  --env_cfg_type aloha-agilex \
  --policy_name Being_H05 \
  --port 6000 \
  --eval_batch false
```

如果要测 batch：

```bash
python XPolicyLab/debug_env_client.py \
  --task_name adjust_bottle \
  --env_cfg_type aloha-agilex \
  --policy_name Being_H05 \
  --port 6000 \
  --eval_batch true
```

说明：

- 这里的 `debug_env_client.py` 默认输入是 mock observation，不是真实 benchmark env
- 因此这一步主要用于接口联调，不代表真实任务成功率

## 训练步骤（XPolicyLab 标准流程）

入口在 `policy/Being_H05/` 顶层脚本，与仓库 README 中 DP / GO1 约定一致：

| 脚本 | 参数个数 | 作用 |
|------|----------|------|
| `process_data.sh` | 5 | 将 LeRobot v2.1 链到 `data/<5-tuple>/` 并注册数据集 |
| `train.sh` | 7 | 训练，权重写到 `checkpoints/<6-tuple>/` |
| `eval.sh` | 11 | 启动 policy server + 环境 client 联调 |

**命名：**

- 处理后数据（5 元组）：`<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`
- 训练产物（6 元组）：上述 5 元组 + `-<seed>`

### 1. 准备 LeRobot 数据

共享 v2.1（14 维 joint，三相机 `cam_*`）：

`/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21`

HDF5 → LeRobot 请用 `XPolicyLab/scripts/transform_lerobot_v30_format.py`（或已有 v21 导出）。**不要用 v30**（缺少 `meta/episodes.jsonl`，当前 Being-H loader 不支持）。

### 2. process_data（5 参数）

```bash
cd XPolicyLab/policy/Being_H05

# 示例：cotrain，3500 条 joint
bash process_data.sh RoboDojo cotrain arx_x5 3500 joint

# 可选：指定其它 LeRobot 根目录
# LEROBOT_DATA_PATH=/path/to/your_lerobot_v21 bash process_data.sh ...
```

输出：`data/RoboDojo-cotrain-arx_x5-3500-joint/`（symlink）及 `Being-H/configs/posttrain/xpolicylab/<5-tuple>.yaml`。

### 3. train（7 参数）

```bash
export BEINGH_MLLM_PATH=/path/to/InternVL3_5-2B
export BEINGH_EXPERT_PATH=/path/to/Qwen3-0.6B
export BEINGH_RESUME_PATH=/path/to/Being-H05-2B

bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3
```

Checkpoint 目录：`checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/`，步数子目录如 `0150000/`。

`data_config` 为 `robodojo_qpos`（非 `robotwin`）。超参可用环境变量覆盖：`MAX_STEPS`、`SAVE_STEPS`、`LEARNING_RATE` 等。

### 4. 评测 / 部署

```bash
bash eval.sh RoboDojo <task_name> cotrain arx_x5 3500 joint 0 <policy_gpu> <env_gpu> beingh <eval_env> 
```

或手动启动 server 时，`model_path` 可指向 6 元组目录或最新 step 子目录；也可只传 6 元组字段由 `model.py` 自动解析。

### RoboTwin（非 RoboDojo）

仍使用 `Being-H/scripts/train/train_robotwin_example.sh` 及 `robotwin_*` 配置，不走上述 5/7 参数流程。




因为flash attn包版本不匹配会导致无法使用torch.compile编译加速，若发现此问题请参考下面conda list替换对应pkg：

```bash
# packages in environment at /root/miniforge3/envs/beingh:
#
# Name                       Version             Build                         Channel
_openmp_mutex                4.5                 7_kmp_llvm                    conda-forge
absl-py                      2.4.0               pypi_0                        pypi
accelerate                   1.11.0              pypi_0                        pypi
addict                       2.4.0               pypi_0                        pypi
aiohappyeyeballs             2.6.1               pypi_0                        pypi
aiohttp                      3.13.3              pypi_0                        pypi
aiosignal                    1.4.0               pypi_0                        pypi
albucore                     0.0.24              pypi_0                        pypi
albumentations               2.0.8               pypi_0                        pypi
annotated-types              0.7.0               pypi_0                        pypi
anyio                        4.13.0              pypi_0                        pypi
asciitree                    0.3.3               pypi_0                        pypi
asttokens                    3.0.1               pypi_0                        pypi
async-timeout                5.0.1               pypi_0                        pypi
attrs                        26.1.0              pypi_0                        pypi
av                           17.0.0              pypi_0                        pypi
blas                         2.116               mkl                           conda-forge
blas-devel                   3.9.0               16_linux64_mkl                conda-forge
blinker                      1.9.0               pypi_0                        pypi
brotli-python                1.2.0               py310hba01987_1               conda-forge
bzip2                        1.0.8               hda65f42_9                    conda-forge
ca-certificates              2026.2.25           hbd8a1cb_0                    conda-forge
certifi                      2026.2.25           pyhd8ed1ab_0                  conda-forge
cffi                         2.0.0               py310he7384ee_1               conda-forge
chardet                      7.3.0               pypi_0                        pypi
charset-normalizer           3.4.6               pyhd8ed1ab_0                  conda-forge
click                        8.3.1               pypi_0                        pypi
cloudpickle                  3.1.2               pypi_0                        pypi
colorlog                     6.10.1              pypi_0                        pypi
comm                         0.2.3               pypi_0                        pypi
configargparse               1.7.5               pypi_0                        pypi
contourpy                    1.3.2               pypi_0                        pypi
cpython                      3.10.20             py310hd8ed1ab_0               conda-forge
cuda-bindings                13.2.0              pypi_0                        pypi
cuda-cudart                  12.4.127            0                             nvidia
cuda-cupti                   12.4.127            0                             nvidia
cuda-libraries               12.4.1              0                             nvidia
cuda-nvrtc                   12.4.127            0                             nvidia
cuda-nvtx                    12.4.127            0                             nvidia
cuda-opencl                  12.9.19             0                             nvidia
cuda-pathfinder              1.5.0               pypi_0                        pypi
cuda-runtime                 12.4.1              0                             nvidia
cuda-toolkit                 13.0.2              pypi_0                        pypi
cuda-version                 12.9                3                             nvidia
cycler                       0.12.1              pypi_0                        pypi
dash                         4.1.0               pypi_0                        pypi
datasets                     4.8.4               pypi_0                        pypi
decorator                    5.2.1               pypi_0                        pypi
decord                       0.6.0               pypi_0                        pypi
deepspeed                    0.18.2              pypi_0                        pypi
dill                         0.4.1               pypi_0                        pypi
einops                       0.8.2               pypi_0                        pypi
embreex                      2.17.7.post7        pypi_0                        pypi
exceptiongroup               1.3.1               pypi_0                        pypi
executing                    2.2.1               pypi_0                        pypi
farama-notifications         0.0.4               pypi_0                        pypi
fasteners                    0.20                pypi_0                        pypi
fastjsonschema               2.21.2              pypi_0                        pypi
ffmpeg                       4.3                 hf484d3e_0                    pytorch
filelock                     3.25.2              pyhd8ed1ab_0                  conda-forge
flash-attn                   2.7.4.post1         pypi_0                        pypi
flask                        3.1.3               pypi_0                        pypi
fonttools                    4.62.1              pypi_0                        pypi
freetype                     2.12.1              h267a509_2                    conda-forge
frozenlist                   1.8.0               pypi_0                        pypi
fsspec                       2026.2.0            pypi_0                        pypi
fvcore                       0.1.5.post20221221  pypi_0                        pypi
giflib                       5.2.2               hd590300_0                    conda-forge
gitdb                        4.0.12              pypi_0                        pypi
gitpython                    3.1.46              pypi_0                        pypi
gmp                          6.3.0               hac33072_2                    conda-forge
gmpy2                        2.3.0               py310h63ebcad_1               conda-forge
gnutls                       3.6.13              h85f3911_1                    conda-forge
grpcio                       1.78.0              pypi_0                        pypi
gymnasium                    0.29.1              pypi_0                        pypi
h11                          0.16.0              pypi_0                        pypi
h2                           4.3.0               pyhcf101f3_0                  conda-forge
h5py                         3.16.0              pypi_0                        pypi
hf-xet                       1.4.2               pypi_0                        pypi
hjson                        3.1.0               pypi_0                        pypi
hpack                        4.1.0               pyhd8ed1ab_0                  conda-forge
httpcore                     1.0.9               pypi_0                        pypi
httpx                        0.28.1              pypi_0                        pypi
huggingface-hub              0.36.2              pypi_0                        pypi
hyperframe                   6.1.0               pyhd8ed1ab_0                  conda-forge
icu                          78.3                h33c6efd_0                    conda-forge
idna                         3.11                pyhd8ed1ab_0                  conda-forge
imageio                      2.37.3              pypi_0                        pypi
imageio-ffmpeg               0.6.0               pypi_0                        pypi
importlib-metadata           9.0.0               pypi_0                        pypi
importlib-resources          6.5.2               pypi_0                        pypi
iopath                       0.1.10              pypi_0                        pypi
ipython                      8.38.0              pypi_0                        pypi
ipywidgets                   8.1.8               pypi_0                        pypi
itsdangerous                 2.2.0               pypi_0                        pypi
jedi                         0.19.2              pypi_0                        pypi
jinja2                       3.1.6               pyhcf101f3_1                  conda-forge
joblib                       1.5.3               pypi_0                        pypi
jpeg                         9e                  h166bdaf_2                    conda-forge
jsonschema                   4.26.0              pypi_0                        pypi
jsonschema-specifications    2025.9.1            pypi_0                        pypi
jupyter-core                 5.9.1               pypi_0                        pypi
jupyterlab-widgets           3.0.16              pypi_0                        pypi
kiwisolver                   1.5.0               pypi_0                        pypi
lame                         3.100               h166bdaf_1003                 conda-forge
lazy-loader                  0.5                 pypi_0                        pypi
lcms2                        2.15                hfd0df8a_0                    conda-forge
ld_impl_linux-64             2.45.1              bootstrap_ha15bf96_2          conda-forge
lerc                         4.1.0               hdb68285_0                    conda-forge
libblas                      3.9.0               16_linux64_mkl                conda-forge
libcblas                     3.9.0               16_linux64_mkl                conda-forge
libcublas                    12.4.5.8            0                             nvidia
libcufft                     11.2.1.3            0                             nvidia
libcufile                    1.14.1.1            4                             nvidia
libcurand                    10.3.10.19          0                             nvidia
libcusolver                  11.6.1.9            0                             nvidia
libcusparse                  12.3.1.170          0                             nvidia
libdeflate                   1.17                h0b41bf4_0                    conda-forge
libexpat                     2.7.4               hecca717_0                    conda-forge
libffi                       3.5.2               h3435931_0                    conda-forge
libgcc                       15.2.0              he0feb66_18                   conda-forge
libgcc-ng                    15.2.0              h69a702a_18                   conda-forge
libgfortran                  15.2.0              h69a702a_18                   conda-forge
libgfortran-ng               15.2.0              h69a702a_18                   conda-forge
libgfortran5                 15.2.0              h68bc16d_18                   conda-forge
libgomp                      15.2.0              he0feb66_18                   conda-forge
libiconv                     1.18                h3b78370_2                    conda-forge
libjpeg-turbo                2.0.0               h9bf148f_0                    pytorch
liblapack                    3.9.0               16_linux64_mkl                conda-forge
liblapacke                   3.9.0               16_linux64_mkl                conda-forge
liblzma                      5.8.2               hb03c661_0                    conda-forge
liblzma-devel                5.8.2               hb03c661_0                    conda-forge
libnpp                       12.2.5.30           0                             nvidia
libnsl                       2.0.1               hb9d3cd8_1                    conda-forge
libnvfatbin                  12.9.82             0                             nvidia
libnvjitlink                 12.4.127            0                             nvidia
libnvjpeg                    12.3.1.117          0                             nvidia
libpng                       1.6.43              h2797004_0                    conda-forge
libsqlite                    3.46.0              hde9e2c9_0                    conda-forge
libstdcxx                    15.2.0              h934c35e_18                   conda-forge
libstdcxx-ng                 15.2.0              hdf11a46_18                   conda-forge
libtiff                      4.5.0               h6adf6a1_2                    conda-forge
libuuid                      2.41.3              h5347b49_0                    conda-forge
libwebp                      1.2.4               h1daa5a0_1                    conda-forge
libwebp-base                 1.2.4               h166bdaf_0                    conda-forge
libxcb                       1.13                h7f98852_1004                 conda-forge
libxcrypt                    4.4.36              hd590300_1                    conda-forge
libzlib                      1.2.13              h4ab18f5_6                    conda-forge
llvm-openmp                  15.0.7              h0cdce71_0                    conda-forge
lxml                         6.0.2               pypi_0                        pypi
manifold3d                   3.4.1               pypi_0                        pypi
markdown                     3.10.2              pypi_0                        pypi
markupsafe                   3.0.3               py310h3406613_1               conda-forge
matplotlib                   3.10.8              pypi_0                        pypi
matplotlib-inline            0.2.1               pypi_0                        pypi
mkl                          2022.1.0            h84fe81f_915                  conda-forge
mkl-devel                    2022.1.0            ha770c72_916                  conda-forge
mkl-include                  2022.1.0            h84fe81f_915                  conda-forge
moviepy                      2.2.1               pypi_0                        pypi
mpc                          1.3.1               h24ddda3_1                    conda-forge
mpfr                         4.2.2               he0a73b1_0                    conda-forge
mplib                        0.2.1               pypi_0                        pypi
mpmath                       1.3.0               pypi_0                        pypi
msgpack                      1.1.2               pypi_0                        pypi
multidict                    6.7.1               pypi_0                        pypi
multiprocess                 0.70.19             pypi_0                        pypi
narwhals                     2.18.1              pypi_0                        pypi
nbformat                     5.10.4              pypi_0                        pypi
ncurses                      6.5                 h2d0b736_3                    conda-forge
nest-asyncio                 1.6.0               pypi_0                        pypi
nettle                       3.6                 he412f7d_0                    conda-forge
networkx                     3.4.2               pyh267e887_2                  conda-forge
ninja                        1.13.0              pypi_0                        pypi
numcodecs                    0.13.1              pypi_0                        pypi
numpy                        1.26.4              pypi_0                        pypi
numpy-quaternion             2024.0.13           pypi_0                        pypi
numpydantic                  1.8.0               pypi_0                        pypi
nvidia-cublas                13.1.0.3            pypi_0                        pypi
nvidia-cublas-cu12           12.4.5.8            pypi_0                        pypi
nvidia-cuda-cupti            13.0.85             pypi_0                        pypi
nvidia-cuda-cupti-cu12       12.4.127            pypi_0                        pypi
nvidia-cuda-nvrtc            13.0.88             pypi_0                        pypi
nvidia-cuda-nvrtc-cu12       12.4.127            pypi_0                        pypi
nvidia-cuda-runtime          13.0.96             pypi_0                        pypi
nvidia-cuda-runtime-cu12     12.4.127            pypi_0                        pypi
nvidia-cudnn-cu12            9.1.0.70            pypi_0                        pypi
nvidia-cufft                 12.0.0.61           pypi_0                        pypi
nvidia-cufft-cu12            11.2.1.3            pypi_0                        pypi
nvidia-cufile                1.15.1.6            pypi_0                        pypi
nvidia-curand                10.4.0.35           pypi_0                        pypi
nvidia-curand-cu12           10.3.5.147          pypi_0                        pypi
nvidia-curobo                0.0.post1.dev1      pypi_0                        pypi
nvidia-cusolver              12.0.4.66           pypi_0                        pypi
nvidia-cusolver-cu12         11.6.1.9            pypi_0                        pypi
nvidia-cusparse              12.6.3.3            pypi_0                        pypi
nvidia-cusparse-cu12         12.3.1.170          pypi_0                        pypi
nvidia-cusparselt-cu12       0.6.2               pypi_0                        pypi
nvidia-nccl-cu12             2.21.5              pypi_0                        pypi
nvidia-nvjitlink             13.0.88             pypi_0                        pypi
nvidia-nvjitlink-cu12        12.4.127            pypi_0                        pypi
nvidia-nvtx                  13.0.85             pypi_0                        pypi
nvidia-nvtx-cu12             12.4.127            pypi_0                        pypi
ocl-icd                      2.3.3               hb9d3cd8_0                    conda-forge
open3d                       0.18.0              pypi_0                        pypi
opencl-headers               2025.06.13          hecca717_0                    conda-forge
opencv-python                4.12.0.88           pypi_0                        pypi
opencv-python-headless       4.13.0.92           pypi_0                        pypi
openh264                     2.1.1               h780b84a_0                    conda-forge
openjpeg                     2.5.0               hfec8fc6_2                    conda-forge
openssl                      3.6.1               h35e630c_1                    conda-forge
packaging                    26.0                pyhcf101f3_0                  conda-forge
pandas                       2.3.3               pypi_0                        pypi
parso                        0.8.6               pypi_0                        pypi
pexpect                      4.9.0               pypi_0                        pypi
pillow                       12.1.1              pypi_0                        pypi
pip                          26.0.1              pyh8b19718_0                  conda-forge
pipablepytorch3d             0.7.6               pypi_0                        pypi
platformdirs                 4.9.4               pypi_0                        pypi
plotly                       6.6.0               pypi_0                        pypi
portalocker                  3.2.0               pypi_0                        pypi
proglog                      0.1.12              pypi_0                        pypi
prompt-toolkit               3.0.52              pypi_0                        pypi
propcache                    0.4.1               pypi_0                        pypi
protobuf                     6.33.6              pypi_0                        pypi
psutil                       7.2.2               pypi_0                        pypi
pthread-stubs                0.4                 hb9d3cd8_1002                 conda-forge
ptyprocess                   0.7.0               pypi_0                        pypi
pure-eval                    0.2.3               pypi_0                        pypi
py-cpuinfo                   9.0.0               pypi_0                        pypi
pyarrow                      23.0.1              pypi_0                        pypi
pybind11                     3.0.2               pypi_0                        pypi
pycollada                    0.9.3               pypi_0                        pypi
pycparser                    2.22                pyh29332c3_1                  conda-forge
pydantic                     2.12.5              pypi_0                        pypi
pydantic-core                2.41.5              pypi_0                        pypi
pyglet                       1.5.31              pypi_0                        pypi
pygments                     2.19.2              pypi_0                        pypi
pyparsing                    3.3.2               pypi_0                        pypi
pyperclip                    1.11.0              pypi_0                        pypi
pyquaternion                 0.9.9               pypi_0                        pypi
pysocks                      1.7.1               pyha55dd90_7                  conda-forge
python                       3.10.14             hd12c33a_0_cpython            conda-forge
python-dateutil              2.9.0.post0         pypi_0                        pypi
python-dotenv                1.2.2               pypi_0                        pypi
python_abi                   3.10                8_cp310                       conda-forge
pytorch                      2.5.1               py3.10_cuda12.4_cudnn9.1.0_0  pytorch
pytorch-cuda                 12.4                hc786d27_7                    pytorch
pytorch-mutex                1.0                 cuda                          pytorch
pytz                         2026.1.post1        pypi_0                        pypi
pyyaml                       6.0.3               py310h3406613_1               conda-forge
pyzmq                        27.1.0              pypi_0                        pypi
readline                     8.3                 h853b02a_0                    conda-forge
referencing                  0.37.0              pypi_0                        pypi
regex                        2026.2.28           pypi_0                        pypi
requests                     2.33.0              pyhcf101f3_0                  conda-forge
retrying                     1.4.2               pypi_0                        pypi
rpds-py                      0.30.0              pypi_0                        pypi
rtree                        1.4.1               pypi_0                        pypi
safetensors                  0.6.2               pypi_0                        pypi
sapien                       3.0.0b1             pypi_0                        pypi
scikit-image                 0.25.2              pypi_0                        pypi
scikit-learn                 1.7.2               pypi_0                        pypi
scipy                        1.15.3              pypi_0                        pypi
sentry-sdk                   2.56.0              pypi_0                        pypi
setuptools                   81.0.0              pypi_0                        pypi
setuptools-scm               10.0.3              pypi_0                        pypi
shapely                      2.1.2               pypi_0                        pypi
simsimd                      6.5.16              pypi_0                        pypi
six                          1.17.0              pypi_0                        pypi
smmap                        5.0.3               pypi_0                        pypi
stack-data                   0.6.3               pypi_0                        pypi
stringzilla                  4.6.0               pypi_0                        pypi
svg-path                     7.0                 pypi_0                        pypi
sympy                        1.13.1              pypi_0                        pypi
tabulate                     0.10.0              pypi_0                        pypi
tbb                          2021.7.0            h924138e_0                    conda-forge
tensorboard                  2.20.0              pypi_0                        pypi
tensorboard-data-server      0.7.2               pypi_0                        pypi
termcolor                    3.3.0               pypi_0                        pypi
threadpoolctl                3.6.0               pypi_0                        pypi
tifffile                     2025.5.10           pypi_0                        pypi
timm                         1.0.26              pypi_0                        pypi
tk                           8.6.13              noxft_h4845f30_101            conda-forge
tokenizers                   0.22.2              pypi_0                        pypi
tomli                        2.4.1               pypi_0                        pypi
toppra                       0.6.3               pypi_0                        pypi
torch                        2.6.0+cu124         pypi_0                        pypi
torchaudio                   2.5.1               py310_cu124                   pytorch
torchtriton                  3.1.0               py310                         pytorch
torchvision                  0.21.0+cu124        pypi_0                        pypi
tqdm                         4.67.3              pypi_0                        pypi
traitlets                    5.14.3              pypi_0                        pypi
transformers                 4.57.1              pypi_0                        pypi
transforms3d                 0.4.2               pypi_0                        pypi
trimesh                      4.4.3               pypi_0                        pypi
triton                       3.2.0               pypi_0                        pypi
typing-inspection            0.4.2               pypi_0                        pypi
typing_extensions            4.15.0              pyhcf101f3_0                  conda-forge
tzdata                       2025.3              pypi_0                        pypi
urllib3                      2.6.3               pypi_0                        pypi
vcs-versioning               1.0.1               pypi_0                        pypi
vhacdx                       0.0.10              pypi_0                        pypi
wandb                        0.25.1              pypi_0                        pypi
warp-lang                    1.12.0              pypi_0                        pypi
wcwidth                      0.6.0               pypi_0                        pypi
werkzeug                     3.1.7               pypi_0                        pypi
wheel                        0.46.3              pyhd8ed1ab_0                  conda-forge
widgetsnbextension           4.0.15              pypi_0                        pypi
xatlas                       0.0.11              pypi_0                        pypi
xorg-libxau                  1.0.12              hb03c661_1                    conda-forge
xorg-libxdmcp                1.1.5               hb03c661_1                    conda-forge
xxhash                       3.6.0               pypi_0                        pypi
xz                           5.8.2               ha02ee65_0                    conda-forge
xz-gpl-tools                 5.8.2               ha02ee65_0                    conda-forge
xz-tools                     5.8.2               hb03c661_0                    conda-forge
yacs                         0.1.8               pypi_0                        pypi
yaml                         0.2.5               h280c20c_3                    conda-forge
yarl                         1.23.0              pypi_0                        pypi
yourdfpy                     0.0.60              pypi_0                        pypi
zarr                         2.18.3              pypi_0                        pypi
zipp                         3.23.0              pypi_0                        pypi
zlib                         1.2.13              h4ab18f5_6                    conda-forge
zstandard                    0.23.0              py310h7c4b9e2_3               conda-forge
zstd                         1.5.6               ha6fb4c9_0                    conda-forge
```