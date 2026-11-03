"""Compatibility entrypoint for the shared owned RGB inference worker."""

from .rgb_service import RGBService as DepthService  # noqa: F401
from .rgb_service import image, verify, worker  # noqa: F401

if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
