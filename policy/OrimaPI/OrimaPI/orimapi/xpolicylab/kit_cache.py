"""Share prepared extensions while isolating each Kit process's writable home."""

from pathlib import Path


def isolated_kit_environment(environment, directory):
    env = dict(environment)
    original_home = Path(env.get("HOME") or Path.home())
    shared_data = Path(env.get("XDG_DATA_HOME") or original_home / ".local/share")
    extensions = shared_data / "ov/data/exts"
    extensions.mkdir(parents=True, exist_ok=True)
    home = Path(directory).resolve()
    link = home / ".local/share/ov/data/exts"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(extensions.resolve(), target_is_directory=True)
    if link.resolve() != extensions.resolve():
        raise ValueError(
            f"Kit extension cache points to an unexpected directory: {link}"
        )
    env.update(
        HOME=str(home),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local/share"),
        ORIMAPI_KIT_REGISTRY_CACHE=str(shared_data / "ov/data/orimapi-kit-registry"),
    )
    return env


def kit_cache_arguments(environment):
    """Override SimulationApp's default portable root in its shared installation."""
    home = Path(environment["HOME"])
    # Keep Kit's V2 registry discovery: registryCacheFull is a different, flat
    # extension layout and must not point at the hashed V2 cache directories.
    return (
        f"--portable-root {home / 'kit'} "
        f"--/app/extensions/registryCache={environment['ORIMAPI_KIT_REGISTRY_CACHE']}"
    )


def canonicalize_registry_links(directory):
    """Keep the shared index usable even after a preparation home is removed."""
    for link in Path(directory).rglob("*"):
        if link.is_symlink():
            target = link.resolve(strict=True)
            if link.readlink() != target:
                replacement = link.with_name(link.name + ".canonical")
                replacement.symlink_to(target, target_is_directory=target.is_dir())
                replacement.replace(link)
