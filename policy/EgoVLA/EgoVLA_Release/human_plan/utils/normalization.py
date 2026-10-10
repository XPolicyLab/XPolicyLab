import torch

# def stats_to_device(
#   ,device,
# ):
  

def normalize_item(
  source_data, normalization_stats
):
  return (
    torch.clamp(
      source_data,
      normalization_stats["lower_bound"].to(source_data.dtype).to(source_data.device),
      normalization_stats["upper_bound"].to(source_data.dtype).to(source_data.device)
    ) - normalization_stats["mean"].to(source_data.dtype).to(source_data.device)
  ) / (
    normalization_stats["std"].to(source_data.dtype).to(source_data.device) + 1e-6
  )


def denormalize_item(
  source_data, normalization_stats
):
  return source_data * (
    normalization_stats["std"].to(source_data.dtype).to(source_data.device) + 1e-6
  ) +  normalization_stats["mean"].to(source_data.dtype).to(source_data.device)
