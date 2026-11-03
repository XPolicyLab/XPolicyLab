"""KnowinAgent: a Dual-arm Asynchronous Self-improving Agent harness.

A Task-Level Agent and a Local Interaction Agent share one Robot Runtime and a
skill library. Only one agent reasons at a time; robot motion on the two arms
runs concurrently through the runtime executor.
"""

__version__ = "1.0.0"
