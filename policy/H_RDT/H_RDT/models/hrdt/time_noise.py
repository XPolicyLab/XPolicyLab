import torch


def logit_normal_sampler(m, s=1, beta_m=100, sample_num=1000000):
    """
    Sampler from the logit-normal distribution.
    """
    y_samples = torch.randn(sample_num) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples


def mu_t(t, a=5, mu_max=4):
    """
    The mu(t) function
    """
    t = t.to('cpu')
    return 2 * mu_max * t ** a - mu_max


def get_beta_s(t, a=5, beta_m=100):
    """
    Get the beta_s for the logit-normal distribution.
    """
    mu = mu_t(t, a=a)
    sigma_s = logit_normal_sampler(m=mu, beta_m=beta_m, sample_num=t.shape[0])
    return sigma_s