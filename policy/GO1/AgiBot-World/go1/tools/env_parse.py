import os


def get_bool_env(name, default=False):
    value = os.environ.get(name, str(default).lower())
    return value.lower() in ("true", "yes", "y", "1")
