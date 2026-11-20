"""WAM inference exports; importing the policy never registers training jobs."""
from .data.data_config import EmbodimentId
from .models.model import SkyrobotWAM

__all__ = ['SkyrobotWAM', 'EmbodimentId']
