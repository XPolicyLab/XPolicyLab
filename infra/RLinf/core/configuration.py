"""Compose policy recipes with shared model and simulator config groups."""

from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from . import extensions


def config_dirs():
    yield from sorted(extensions.POLICIES.glob("*/rlinf/config"))
    yield from sorted((extensions.ADAPTER / "envs").glob("*/config"))


class AdapterSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        for directory in config_dirs():
            search_path.append(provider="xpolicylab", path=directory.as_uri())


def register_config_paths():
    Plugins.instance().register(AdapterSearchPath)
