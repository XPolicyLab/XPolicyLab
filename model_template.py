class ModelTemplate:
    """
    Base template for a policy/model class.
    策略/模型类的基础模板。

    This template defines the minimal interface required for:
    该模板定义了一个模型在以下场景中常用的最小接口：
    1. loading or building a model
       加载或构建模型
    2. updating observations from the environment
       接收并更新环境观测
    3. generating actions
       生成动作
    4. resetting internal states between episodes
       在每个 episode 之间重置内部状态

    Notes / 说明:
    - You should override the methods in your own subclass or implementation.
      你应该在自己的子类或具体实现中重写这些方法。
    - `self.model` can be a neural network, a planner, or any policy object.
      `self.model` 可以是神经网络、规划器，或者任意策略对象。
    """

    RED = "\033[31m"
    RESET = "\033[0m"

    def __init__(self):
        """
        Initialize the template model.
        初始化模板模型。

        Attributes / 属性:
            self.model:
                The actual model/policy object.
                实际的模型/策略对象。
        """
        self.model = None

    def _error_msg(self, msg: str) -> str:
        return f"{self.RED}[ERROR] {msg}{self.RESET}"

    def update_obs(self, obs):
        """
        Update the current observation used by the model.
        更新当前模型使用的观测。

        The policy server decodes camera colors before calling this, so
        obs["vision"][<camera>]["color"] is always an image array.
        策略服务端已完成图像解码，此处拿到的 color 一定是图像数组。
        """
        raise NotImplementedError(
            self._error_msg("update_obs() must be implemented by the user.")
        )

    def update_obs_batch(self, obs_list):
        """
        Update the current observation used by the model.
        更新当前模型使用的观测。

        Same decoding guarantee as update_obs().
        与 update_obs() 一样，图像已由服务端解码。
        """
        raise NotImplementedError(
            self._error_msg("update_obs_batch() must be implemented by the user.")
        )

    def get_action(self):
        """
        Predict or generate an action from the current observation/state.
        根据当前观测或内部状态预测/生成动作。
        """
        raise NotImplementedError(
            self._error_msg("get_action() must be implemented by the user.")
        )

    def get_action_batch(self, env_idx_list=None):
        """
        Predict or generate an action from the current observation/state.
        根据当前观测或内部状态预测/生成动作。
        """
        raise NotImplementedError(
            self._error_msg("get_action_batch() must be implemented by the user.")
        )

    def reset(self):
        """
        Reset the internal state of the model at the start of a new episode.
        在新 episode 开始时重置模型内部状态。
        """
        pass

    def prepare_case(self, case_meta=None):
        """
        Optional hook called before an action case starts.
        在 action case 开始前调用的 hook。
        """
        pass

    def on_trial_end(self, result=None):
        """
        Optional hook called after a trial finishes.
        在单次 trial 结束后调用的 hook。
        """
        pass

    # ---- Optional RL training contract (used by RLinf) ----
    # 可选的 RL 训练接口（RLinf 使用）
    #
    # RL_MODES lists implemented capabilities: eval, sft, ppo, grpo, nft.
    # deploy/train config ``rl.modes`` picks what one run enables (default: eval),
    # so plain serving never loads training-only code.
    # RL_MODES 声明策略实现了哪些模式；配置项 ``rl.modes`` 决定本次运行启用哪些（默认仅 eval）。
    RL_MODES = ("eval",)

    def get_rl_modes(self):
        """
        Return the RL modes enabled by ``model_cfg["rl"]``.
        返回 ``model_cfg["rl"]`` 启用的 RL 模式。
        """
        model_cfg = getattr(self, "model_cfg", None) or {}
        rl_cfg = model_cfg.get("rl") or {}
        if rl_cfg.get("enabled") is False:
            return ("eval",)
        modes = tuple(str(mode).strip().lower() for mode in rl_cfg.get("modes", ("eval",)))
        unsupported = [mode for mode in modes if mode not in self.RL_MODES]
        if unsupported:
            raise ValueError(
                self._error_msg(f"rl.modes {unsupported} not in {type(self).__name__}.RL_MODES={self.RL_MODES}.")
            )
        return modes

    def predict_action_batch_rl(self, obs_list, mode="train", compute_values=True, **kwargs):
        """
        Sample action chunks for RL rollout.
        为 RL rollout 采样动作块。

        Args:
            obs_list: XPolicyLab observations, same format as update_obs_batch().
                与 update_obs_batch() 相同格式的观测列表。
            mode: "train" samples stochastically and records log-probs; "eval" is deterministic.
                "train" 随机采样并记录 log-prob；"eval" 为确定性推理。
            compute_values: whether to return value estimates (PPO).
                是否返回价值估计（PPO 需要）。

        Returns:
            actions: ``[B, H, D]`` array packed in ``pack_robot_state`` order.
                按 ``pack_robot_state`` 顺序打包的动作。
            extras: dict with
                ``prev_logprobs`` (tensor, required for ppo/grpo),
                ``prev_values`` (tensor ``[B, 1]``, required for ppo),
                ``forward_inputs`` (dict of batch-first tensors, everything
                compute_logprob_value() needs to recompute the same sample).
        """
        raise NotImplementedError(
            self._error_msg("predict_action_batch_rl() is not implemented.")
        )

    def compute_logprob_value(self, forward_inputs, compute_values=False, **kwargs):
        """
        Recompute log-probs (and values) of recorded rollout samples, with gradients.
        对 rollout 样本重新计算 log-prob（及价值），需保留梯度。

        Returns a dict with ``logprobs`` (same shape as ``prev_logprobs``),
        ``values`` (``[B]``) and ``entropy`` (``[B, 1]``).
        Never fake a log-prob: leave ppo/grpo out of RL_MODES instead.
        不要伪造 log-prob：不支持时不要在 RL_MODES 中声明 ppo/grpo。
        """
        raise NotImplementedError(
            self._error_msg("compute_logprob_value() is not implemented.")
        )

    def sft_forward(self, obs_list, actions=None, **kwargs):
        """
        Return the scalar supervised loss for a batch of observations and target actions.
        返回一批观测与目标动作的监督学习标量损失。
        """
        raise NotImplementedError(
            self._error_msg("sft_forward() is not implemented.")
        )

    def nft_forward(self, forward_inputs, **kwargs):
        """Recompute recorded denoising predictions for an NFT objective.

        重算 NFT 样本的去噪预测；此接口不声明 PPO/GRPO 概率能力。
        """
        raise NotImplementedError(self._error_msg("nft_forward() is not implemented."))
