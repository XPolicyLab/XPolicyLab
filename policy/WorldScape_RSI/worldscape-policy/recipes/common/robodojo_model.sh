#!/usr/bin/env bash
# Select the same public YAML for data preparation and training.
export ROBODOJO_MODEL="${ROBODOJO_MODEL:-flash}"
case "$ROBODOJO_MODEL" in
    flash) robodojo_config_name=robodojo.yaml ;;
    memory) robodojo_config_name=robodojo_memory.yaml ;;
    *)
        echo "ERROR: ROBODOJO_MODEL must be flash or memory, got '$ROBODOJO_MODEL'" >&2
        exit 2
        ;;
esac
