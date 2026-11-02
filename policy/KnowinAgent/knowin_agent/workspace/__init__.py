"""Per-episode workspaces: layout, include resolution, phase permissions, freezing and library staging."""

from knowin_agent.workspace.freeze import FrozenChanged, freeze, freeze_dirs, load_manifest, verify
from knowin_agent.workspace.includes import IncludeError, register_builtin, resolve_includes
from knowin_agent.workspace.knowledge import (LIBRARY_MAX_BYTES, LIBRARY_MAX_FILES, check_library, latest_published,
                                      library_dir, library_root, load_versions, register_version,
                                      seed_knowledge, stage_library)
from knowin_agent.workspace.layout import LESSON_FILES, Workspace, create_workspace, open_workspace
from knowin_agent.workspace.permissions import apply_permissions, assert_writable

__all__ = [
    "FrozenChanged", "IncludeError", "LESSON_FILES", "LIBRARY_MAX_BYTES", "LIBRARY_MAX_FILES", "Workspace",
    "apply_permissions", "assert_writable", "check_library", "create_workspace", "freeze", "freeze_dirs",
    "latest_published", "library_dir", "library_root", "load_manifest", "load_versions", "open_workspace",
    "register_builtin", "register_version", "resolve_includes", "seed_knowledge", "stage_library", "verify",
]
