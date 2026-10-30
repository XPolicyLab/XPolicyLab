import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LatentStatsTracker(nn.Module):
    def __init__(self, num_channels=512, window_size=50):
        super().__init__()
        self.window_size = window_size
        self.num_channels = num_channels
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_std", torch.ones(1, num_channels, 1, 1))
        self.register_buffer("step_count", torch.tensor(0))
        self.register_buffer("history_mean", torch.zeros(window_size, num_channels))
        self.register_buffer("history_var", torch.ones(window_size, num_channels))

    def update(self, x):
        with torch.no_grad():
            curr_mean = x.mean(dim=(0, 2, 3))
            curr_var = x.var(dim=(0, 2, 3))
            idx = self.step_count % self.window_size
            self.history_mean[idx] = curr_mean
            self.history_var[idx] = curr_var
            self.step_count += 1
            actual_window = min(self.step_count, self.window_size)
            self.running_mean[0, :, 0, 0] = self.history_mean[:actual_window].mean(dim=0)
            combined_var = self.history_var[:actual_window].mean(dim=0)
            self.running_std[0, :, 0, 0] = torch.sqrt(combined_var + 1e-6)

    def normalize(self, x):
        return (x - self.running_mean) / self.running_std

    def denormalize(self, x):
        return x * self.running_std + self.running_mean