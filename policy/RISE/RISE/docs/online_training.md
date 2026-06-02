
## 🏃‍♂️ Online Training

To start training, run the embodiment script with your configuration file:

```bash
bash policy_and_value/policy_online/examples/embodiment/run_embodiment.sh YOUR_CONFIG_NAME

# Example:
bash policy_and_value/policy_online/examples/embodiment/run_embodiment.sh rl_release

```

*Notes: The initial run may take a while (~10 minutes) as loading the dynamics and reward models, along with torch compile building the acceleration graph, all contribute to the latency. For debugging purposes, you can set actor.model.openpi.use_torch_compile = False.*

### 🖥️ Cluster Configuration

You can flexibly configure the GPU allocation for the `env`, `rollout`, and `actor` components in your YAML config. Here are three common deployment strategies:

* **Partial Sharing (Default):** Components share some GPUs while keeping others dedicated.
```yaml
cluster:
  num_nodes: 1
  component_placement:
    env: 0-3
    rollout: 4-7
    actor: 0-7

```


* **Complete Sharing:** All components share all available GPUs.
```yaml
cluster:
  num_nodes: 1
  component_placement:
    env,rollout,actor: all
```


* **Complete Separation:** Each component uses its own GPUs without interference, eliminating the need for offload functionality.
```yaml
cluster:
  num_nodes: 1
  component_placement:
    env: 0-1
    rollout: 2-5
    actor: 6-7

```



### 🌐 Multi-Node Training

For `N`-node training, change `cluster.num_nodes` to `N` and assign the `component_placement` accordingly. *(e.g., If N=2 and each node has 8 GPUs, the placement indices range from 0 to 15).*

Run the multi-task unified training command:

```bash
bash policy_and_value/policy_online/examples/embodiment/run_embodiment_ray_unified_multi_task.sh YOUR_CONFIG_NAME

```

### 🔄 Resuming from Checkpoint

To resume training, modify `runner.resume_dir` in your config to point to your target checkpoint:

```yaml
runner:
  resume_dir: logs/20251221-00:15:14/${runner.logger.experiment_name}/checkpoints/global_step_13000
```

---

## ⚙️ Configuration Parameters

| Parameter | Description |
| --- | --- |
| `algorithm.num_group_envs` | Number of parallel environments for rollout. *(e.g., If set to 32 with 4 GPUs for rollout, each GPU handles 8 envs).* |
| `algorithm.rollout_epoch` | Number of epochs for rollouts. |
| `algorithm.policy_config_name` | Task-specific configuration. **Must strictly align** with your offline (IL) training setting. |
| `rollout.model_dir` | Path to your pretrained IL model for initialization. |
| `actor.micro_batch_size` | Micro-batch size per GPU. |
| `actor.global_batch_size` | Global batch size across all GPUs. |
| `model.action_dim` | Expected action dimension output for VLA models. |
| `rollout_ema_decay` | EMA preserving weight for each rollout model update. |
| `dynamics_model_config` | Task-specific configuration for the dynamics model. |
| `dynamics_model_image_root` | *(Optional)* Custom path for dynamics model images. |
| `dynamics_model_output_path` | *(Optional)* Custom path for dynamics model outputs. |
| `reward_model_config` | Task-specific configuration for the reward model. |
| `reward_model_ckpt` | Checkpoint path for the reward model. |
| `visualize_wm_pred` | Set to `True` to visualize your world model predictions. If `True`, the `chunk_reward` should be `True` too.|
| `chunk_reward` | Set to `True` to use only the reward of the last predicted frame as the reward for the current action chunk. |
| `advantage_scale` | Weighted coefficient for the computed advantage. |

> **Note:** For other configurations not listed here, we adopt most settings from **RLinf**. Please refer to the [RLinf Documentation](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/yaml.html) for more details.

---

## 📦 Deployment

Once you have trained your own VLA model, you need to convert the Distributed Checkpoint (`.dcp`) to a PyTorch state dict (`.pt`) before deployment.

Run the converter script:

```bash
python toolkits/ckpt_convertor/convert_dcp_to_state_dict.py \
    --dcp_path <YOUR_DCP_CKPT_DIR> \
    --output_path <YOUR_EXPECTED_PT_CKPT_DIR>

```

After conversion, you can use the generated `.pt` checkpoints on your deployment machine to infer actions.
