import math

import torch

# Clamp the uniform draw away from the exact CDF edges so the inverse-CDF
# erfinv() never hits +/-inf when a truncation band touches t=0 or t=1.
_TRUNC_U_EPS = 1e-7


def inverse_shift(y: float, shift: float) -> float:
    """
    Invert the flow-matching schedule shift  s(x) = a*x / (1 + (a-1)*x).

    Given a post-shift value ``y`` in [0, 1], return the raw (pre-shift) value
    ``x`` with ``s(x) == y``. Used to convert an expert's t-space noise band
    into the raw space where the logit-normal draw is truncated (the trainer
    applies the shift *after* this sampler returns). See
    docs/two_expert_two_step_meanflow.md section 4.
    """
    if shift == 1.0:
        return y
    return y / (shift - (shift - 1.0) * y)


def _truncation_cdf_bound(raw: float, mu: float, sigma: float) -> float:
    """
    Standard-normal CDF of ``(logit(raw) - mu) / sigma``: the value of the
    uniform draw ``u`` at which the truncated-logit-normal inverse-CDF maps to
    ``raw``. ``raw <= 0`` -> 0.0 and ``raw >= 1`` -> 1.0 (that side untruncated).
    """
    if raw <= 0.0:
        return 0.0
    if raw >= 1.0:
        return 1.0
    logit = math.log(raw / (1.0 - raw))
    z = (logit - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sample_improved_meanflow_times(
    batch_size: int,
    device: torch.device,
    dist: str = "lognorm",
    mu: float = -0.4,
    sigma: float = 1.0,
    flow_ratio: float = 0.5,
    raw_lo: float = 0.0,
    raw_hi: float = 1.0,
    r_mode: str = "ordered",
    dtype: torch.dtype = None,
    generator: torch.Generator = None,
):
    """
    Sample t, r in [0, 1] for improved MeanFlow.

    - dist: "lognorm" uses logit-normal (sigmoid of normal).
    - flow_ratio: fraction of samples where r = t (flow matching samples).
    - raw_lo, raw_hi: truncate the logit-normal draw to this raw (pre-shift)
      band via an inverse-CDF transform. The default [0.0, 1.0] applies no
      truncation and reproduces the plain logit-normal. Used by the two-expert
      noise-band split (docs/two_expert_two_step_meanflow.md).
    - r_mode: how r is coupled to t.
        "ordered" (default): draw t and r independently from `dist`, then
            sort the pair so t >= r (unchanged historical behaviour).
        "uniform_to_zero": keep the drawn t, then resample r ~ Uniform(0, t)
            per-sample (the independently-drawn r is discarded). This makes
            the interval endpoint r uniform on [0, t] rather than another
            draw from `dist`.

    Returns:
        t, r: tensors of shape (batch_size,) on device.
    """
    dtype = dtype or torch.float32
    if dist == "uniform":
        t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        r = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    elif dist == "lognorm":
        if raw_lo <= 0.0 and raw_hi >= 1.0:
            # No truncation requested: plain logit-normal (unchanged behaviour).
            normal_t = torch.randn(batch_size, device=device, dtype=dtype, generator=generator) * sigma + mu
            normal_r = torch.randn(batch_size, device=device, dtype=dtype, generator=generator) * sigma + mu
            t = torch.sigmoid(normal_t)
            r = torch.sigmoid(normal_r)
        else:
            # Truncated logit-normal via inverse-CDF: draw a uniform inside the
            # band's CDF range, then map back through the normal quantile.
            lo = _truncation_cdf_bound(raw_lo, mu, sigma)
            hi = _truncation_cdf_bound(raw_hi, mu, sigma)
            sqrt2 = math.sqrt(2.0)

            def _draw_truncated() -> torch.Tensor:
                u = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
                u = lo + (hi - lo) * u
                u = u.clamp(_TRUNC_U_EPS, 1.0 - _TRUNC_U_EPS)
                normal = mu + sigma * sqrt2 * torch.erfinv(2.0 * u - 1.0)
                return torch.sigmoid(normal).to(dtype)

            t = _draw_truncated()
            r = _draw_truncated()
    else:
        raise ValueError(f"Unknown time distribution: {dist}")

    if r_mode == "ordered":
        # Independent draws, sorted so t >= r (historical behaviour).
        t, r = torch.maximum(t, r), torch.minimum(t, r)
    elif r_mode == "uniform_to_zero":
        # Keep t as drawn; resample r uniformly on [0, t] per-sample. The
        # independently-drawn r above is discarded in this mode.
        u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        r = u * t
    else:
        raise ValueError(f"Unknown r_mode: {r_mode}")

    if flow_ratio > 0.0:
        num_flow = int(flow_ratio * batch_size)
        if num_flow > 0:
            r = r.clone()
            r[:num_flow] = t[:num_flow]

    return t, r


def adaptive_weighted_loss(loss_per_sample: torch.Tensor, norm_p: float = 1.0, norm_eps: float = 0.01):
    """
    Adaptive weighting: loss / stopgrad((loss + eps)^p)
    """
    weight = (loss_per_sample + norm_eps).pow(norm_p)
    return loss_per_sample / weight.detach()