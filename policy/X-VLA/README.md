# 数据转化
## robotwin数据格式
如果是robotwin格式数据, 直接修改对应路径即可.

```bash
python hdf5_add_language_instruction.py
```

## 其他数据格式
TODO

# 训练
修改`meta.json`, 把所有要勇于训练的数据路径放入.
然后修改`train.sh`, 设置三个参数:  
**models**, **train_metas_path**, and **output_dir**

# XPolicyLab 推理封装
`policy/X-VLA` 已经按 `Pi_05` 风格接入 `XPolicyLab`，但内部不再额外启动 X-VLA 自己的 HTTP server，而是直接在 `Model` 进程内加载权重并推理。

注意：
- 在 `XPolicyLab` 中实际使用的策略名是 `XVLA`，这是为了绕开 `X-VLA` 目录名中的 `-` 无法被 Python import 的限制。
- `action_type` 当前固定使用 `ee`。
- 如果 checkpoint 不包含 processor 文件，请额外传入 `processor_path` 指向 base checkpoint。
