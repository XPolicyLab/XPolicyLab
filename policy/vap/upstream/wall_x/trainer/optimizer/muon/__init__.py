from .muon import Muon, build_muon_optimizer, zeropower_via_newtonschulz5
from .shape_aware_muon import MuonZero0, build_shape_aware_muon_optimizer
from .utils import get_muon_optimizer  # side-effect: registers "muon" dispatcher

__all__ = [
    "Muon",
    "MuonZero0",
    "build_muon_optimizer",
    "build_shape_aware_muon_optimizer",
    "get_muon_optimizer",
    "zeropower_via_newtonschulz5",
]
