"""Resolve and record the approved execution span, independently of VLA horizon."""
import os
import re

MIN_STEPS_ENV = "ROBORSI_ADAPTIVE_MIN_STEPS"


def resolve_adaptive_steps(environ=None):
    """Read a deployment override once; explicit invalid values never fall back."""
    environment = os.environ if environ is None else environ
    value = environment.get(MIN_STEPS_ENV, "5")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,2}", value):
        raise ValueError(f"{MIN_STEPS_ENV} must be an integer from 2 to 50")
    minimum = int(value)
    if not 2 <= minimum <= 50:
        raise ValueError(f"{MIN_STEPS_ENV} must be an integer from 2 to 50")
    return {
        "approved_steps_min": minimum,
        "approved_steps_max": 50,
        "approved_steps_default": max(5, minimum),
    }


def configured_adaptive_steps(options):
    """Use the frozen episode settings, never a changing process environment."""
    minimum = options.get("approved_steps_min", 5)
    maximum = options.get("approved_steps_max", 50)
    default = options.get("approved_steps_default", max(5, minimum) if type(minimum) is int else 5)
    if (type(minimum) is not int or not 2 <= minimum <= 50
            or type(maximum) is not int or maximum != 50
            or type(default) is not int or not minimum <= default <= maximum):
        raise ValueError("Invalid configured adaptive execution limits")
    return {
        "approved_steps_min": minimum,
        "approved_steps_max": maximum,
        "approved_steps_default": default,
    }


def apply_adaptive_steps(cfg):
    """Overlay deployment limits without modifying an authenticated template."""
    return {**cfg, "options": {**cfg.get("options", {}), **resolve_adaptive_steps()}}
