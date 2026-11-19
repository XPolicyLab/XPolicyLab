from pathlib import Path

_work_dir = Path.cwd()


def register_work_dir(path):
    global _work_dir
    _work_dir = Path(path)


def get_work_dir():
    return str(_work_dir)
