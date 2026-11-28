"""Logging used by the inference framework."""
import logging


def initialize_overwatch(name):
    return logging.getLogger(name)
