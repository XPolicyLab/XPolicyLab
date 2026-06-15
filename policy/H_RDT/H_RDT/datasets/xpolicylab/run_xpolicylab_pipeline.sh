#!/bin/bash
set -e

# H-RDT XPolicyLab Data Processing Pipeline

ENABLE_TASK_INSTRUCTION_EXTRACTION=${ENABLE_TASK_INSTRUCTION_EXTRACTION:-false}
ENABLE_STATS_CALCULATION=${ENABLE_STATS_CALCULATION:-true}
ENABLE_LANGUAGE_ENCODING=${ENABLE_LANGUAGE_ENCODING:-false}
XPOLICYLAB_TASKS=${XPOLICYLAB_TASKS:-all}

if [ -z "$XPOLICYLAB_DATA_ROOT" ]; then
    source "$(dirname "$0")/setup_xpolicylab.sh"
fi

cd "$HRDT_PROJECT_ROOT"

export TASK_INSTRUCTIONS_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/task_instructions.csv"
export STATS_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/stats.json"
export LANG_EMBEDDINGS_DIR="${HRDT_OUTPUT_DIR}/lang_embeddings"

echo "Starting XPolicyLab data processing pipeline..."
echo "Data Root: $XPOLICYLAB_DATA_ROOT"
echo "Raw Dataset: $XPOLICYLAB_RAW_DATASET_NAME"
echo "Output Dir: $HRDT_OUTPUT_DIR"
echo "Tasks: $XPOLICYLAB_TASKS"
echo "Task instruction extraction enabled: $ENABLE_TASK_INSTRUCTION_EXTRACTION"
echo "Stats calculation enabled: $ENABLE_STATS_CALCULATION"
echo "Language encoding enabled: $ENABLE_LANGUAGE_ENCODING"

if [ "$ENABLE_TASK_INSTRUCTION_EXTRACTION" = "true" ]; then
    echo "Step 1: Extracting task instructions..."
    python datasets/xpolicylab/extract_task_instructions.py \
        "$XPOLICYLAB_DATA_ROOT" \
        --env_cfg_type "$XPOLICYLAB_ENV_CFG_TYPE" \
        --output "$TASK_INSTRUCTIONS_OUTPUT_PATH"
else
    echo "Step 1: Skipping task instruction extraction"
fi

if [ "$ENABLE_STATS_CALCULATION" = "true" ]; then
    echo "Step 2: Calculating q01/q99 action statistics..."
    max_episode_args=()
    if [ -n "$XPOLICYLAB_MAX_EPISODES" ]; then
        max_episode_args+=(--max_episodes "$XPOLICYLAB_MAX_EPISODES")
    fi

    python datasets/xpolicylab/calc_stat.py \
        --data_root "$XPOLICYLAB_DATA_ROOT" \
        --raw_dataset_name "$XPOLICYLAB_RAW_DATASET_NAME" \
        --env_cfg_type "$XPOLICYLAB_ENV_CFG_TYPE" \
        --action_type "$XPOLICYLAB_ACTION_TYPE" \
        --tasks "$XPOLICYLAB_TASKS" \
        --output_path "$STATS_OUTPUT_PATH" \
        "${max_episode_args[@]}"
else
    echo "Step 2: Skipping stats calculation"
fi

if [ "$ENABLE_LANGUAGE_ENCODING" = "true" ]; then
    echo "Step 3: Encoding language embeddings..."
    python datasets/xpolicylab/encode_lang_batch.py
else
    echo "Step 3: Skipping language encoding"
    if [ -d "$LANG_EMBEDDINGS_DIR" ]; then
        echo "Language embeddings found at: $LANG_EMBEDDINGS_DIR"
    else
        echo "Warning: Language embeddings directory not found: $LANG_EMBEDDINGS_DIR"
        echo "Run with ENABLE_LANGUAGE_ENCODING=true or provide embeddings before training."
    fi
fi

echo ""
echo "XPolicyLab pipeline completed!"
echo "Available files:"
echo "  - Task instructions: $TASK_INSTRUCTIONS_OUTPUT_PATH"
echo "  - Statistics: $STATS_OUTPUT_PATH"
echo "  - Language embeddings: $LANG_EMBEDDINGS_DIR/*.pt"
echo ""
echo "Examples:"
echo "  ENABLE_TASK_INSTRUCTION_EXTRACTION=true ./datasets/xpolicylab/run_xpolicylab_pipeline.sh"
echo "  ENABLE_LANGUAGE_ENCODING=true ./datasets/xpolicylab/run_xpolicylab_pipeline.sh"
