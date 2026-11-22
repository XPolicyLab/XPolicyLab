import torch
import math


class FlowMatchScheduler:

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
        training_weight_floor=1e-6,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.training_weight_floor = training_weight_floor
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
        exponential_shift_mu=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            if exponential_shift_mu is not None:
                mu = exponential_shift_mu
            elif dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            else:
                mu = self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = (
                self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
            )
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            if self.training_weight_floor > 0:
                y_shifted = y_shifted.clamp_min(self.training_weight_floor)
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def _prepare_timestep(self, timestep, op_name):
        if isinstance(timestep, torch.Tensor):
            if timestep.dtype in (torch.float16, torch.bfloat16):
                raise ValueError(
                    f"FlowMatchScheduler.{op_name} received a {timestep.dtype} "
                    "timestep. Pass fp32 scheduler timesteps; converting "
                    "bf16/fp16 back to fp32 cannot recover the lost noise-level precision."
                )
            return timestep.to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        return torch.as_tensor(
            timestep, device=self.timesteps.device, dtype=self.timesteps.dtype
        )

    def step(
        self,
        model_output,
        timestep,
        sample,
        to_final=False,
        pred_type="v_pred",
        **kwargs,
    ):
        timestep = self._prepare_timestep(timestep, "step")
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]

        if pred_type == "x_pred":
            # model_output is x0_pred; convert to velocity: v = (x_t - x0) / sigma
            if sigma > 0:
                velocity = (sample - model_output) / sigma
            else:
                velocity = torch.zeros_like(model_output)
            prev_sample = sample + velocity * (sigma_ - sigma)
        else:
            prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        timestep = self._prepare_timestep(timestep, "return_to_timestep")
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        timestep = self._prepare_timestep(timestep, "add_noise")
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep, pred_type="v_pred"):
        """
        Compute training target based on prediction type.
        Args:
            sample: clean data x_0
            noise: noise epsilon
            timestep: current timestep
            pred_type: "v_pred" for velocity prediction (default, target = noise - x_0),
                       "x_pred" for data prediction (JiT-style, target = x_0).
        """
        if pred_type == "x_pred":
            return sample
        else:
            return noise - sample

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - self._prepare_timestep(timestep, "training_weight")).abs()
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def invert_to_data_domain(
        self, model_pred, noisy_sample, timestep, pred_type="v_pred"
    ):
        """
        Invert from model prediction to data domain.

        For v_pred (velocity prediction):
            x_t = (1 - rho(t)) * x_0 + rho(t) * eps
            Training target: v = eps - x_0
            Therefore: x_hat_0 = x_t - rho(t) * v

        For x_pred (data prediction, JiT-style):
            Model directly predicts x_0, no conversion needed.

        Args:
            model_pred: Model prediction (velocity v for v_pred, x_0 for x_pred)
            noisy_sample: Current noisy sample x_t, shape [..., dim]
            timestep: Current timestep t
            pred_type: "v_pred" or "x_pred"

        Returns:
            x_hat_0: Data domain prediction, shape [..., dim]
        """
        if pred_type == "x_pred":
            # Model directly predicts x_0
            return model_pred

        timestep = self._prepare_timestep(timestep, "invert_to_data_domain")

        # Get sigma value corresponding to current timestep (i.e., rho(t))
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]

        # x̂_0 = x_t - ρ(t) * v
        x0_pred = noisy_sample - sigma * model_pred
        return x0_pred
