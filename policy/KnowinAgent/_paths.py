"""Locate runtime resources in a development checkout or exported policy bundle."""

from pathlib import Path


ADAPTER_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    # The standalone policy owns its resources; the development adapter is two
    # directories below them. Resolve __file__ so a policy symlink also works.
    for root in (ADAPTER_ROOT, ADAPTER_ROOT.parents[1]):
        if (root / "knowin_agent" / "__init__.py").is_file() and (root / "knowledge").is_dir():
            return root
    raise FileNotFoundError(f"KnowinAgent runtime and knowledge not found beside {ADAPTER_ROOT}")


PACKAGE_ROOT = package_root()
KNOWLEDGE = PACKAGE_ROOT / "knowledge"
CONFIG = ADAPTER_ROOT / "test_config.json"


if __name__ == "__main__":
    print(PACKAGE_ROOT)
