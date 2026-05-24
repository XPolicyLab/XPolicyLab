from diffusers.utils.import_utils import _is_package_available

_lerobot_available, _lerobot_version = _is_package_available('lerobot')


def is_lerobot_available() -> bool:
    """Check if the optional dependency 'lerobot' is available in the expected
    version.

    Returns:
        bool: True if the package 'lerobot' is installed and its version equals '0.3.2'; otherwise False.
    """
    if not _lerobot_available:
        return False
    if _lerobot_version != '0.3.2':
        return False
    return True
