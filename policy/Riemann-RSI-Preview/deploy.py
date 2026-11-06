"""Official submission entry; deployment behavior lives in the named adapter."""
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from importlib import import_module

_adapter = import_module('rsi.adapters.Riemann-RSI-Preview.deploy')
eval_one_episode = _adapter.eval_one_episode
eval_one_episode_batch = _adapter.eval_one_episode_batch

__all__ = ['eval_one_episode', 'eval_one_episode_batch']
