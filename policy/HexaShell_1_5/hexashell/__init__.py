"""HexaShell 1.5: a language-model robot agent for RoboDojo's two-arm table.

The model writes Python cells against a small ``robot`` API; the cells run in a sandboxed worker
process, and every motion they ask for is turned into joint targets that RoboDojo's evaluation
client executes one action at a time.

Packages:
  runtime  episode driver, LLM loop, image window, records
  kernel   sandboxed worker process and its RPC link to the host
  robot    the ``robot.*`` API the cells see, and the printed result types
  motion   arm kinematics, inverse kinematics, paths, the action executor
  vision   cameras, SAM3 client, robot mask, multi-view measurement, overlays
  xpl      XPolicyLab adapter (policy server side) and the obs/action bridge
"""

__version__ = "1.5.0"
