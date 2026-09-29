"""Overlay only the CogWAM policy while reusing RoboDojo's own XPolicyLab."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
