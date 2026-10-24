"""The Harness (Harness Policy in the paper): a coding agent as a generalist robot policy. No simulator imports here."""

__version__ = "0.1.0"

# The version of the agent: its high-level loop, prompts, skills and default
# configuration (configs/default.toml). Every lock and episode manifest records it.
# Raise the minor version for a change in what the agent does, the patch version for a
# fix; docs/changelog.md says what changed and the evidence for it.
AGENT_VERSION = "1.10.0"
