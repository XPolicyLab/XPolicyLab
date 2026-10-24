"""Connectors: how a run reaches a robot when it is not a worker the suite starts.

xpolicylab.py   the harness as an XPolicyLab policy server (evaluated by RoboDojo's,
                RoboTwin's or a real robot's own XPolicyLab client); the bridge that
                meets XPolicyLab's calls and the session's ticks
package.py      what an exported policy/<Name>/ folder runs (harness export xpolicylab)
"""
