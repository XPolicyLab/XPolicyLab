#!/usr/bin/env bash
# Shared training bootstrap and base-component checkpoint defaults.

wsp_hf_download() {
    local python_bin_dir
    python_bin_dir="$(dirname -- "$WSP_PYTHON")"
    if [[ -x "$python_bin_dir/hf" ]]; then
        "$python_bin_dir/hf" download "$@"
    elif command -v hf >/dev/null 2>&1; then
        hf download "$@"
    elif [[ -x "$python_bin_dir/huggingface-cli" ]]; then
        "$python_bin_dir/huggingface-cli" download "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$@"
    else
        echo "ERROR: Hugging Face CLI is missing; install huggingface_hub" >&2
        return 1
    fi
}

wsp_train_setup() {
    local root="$1"

    if [[ -z "${WSP_PYTHON:-}" ]]; then
        if [[ -n "${CONDA_ENV:-}" ]]; then
            WSP_PYTHON="$CONDA_ENV/bin/python"
        elif [[ -n "${CONDA_PREFIX:-}" ]]; then
            CONDA_ENV="$CONDA_PREFIX"
            WSP_PYTHON="$CONDA_PREFIX/bin/python"
        else
            WSP_PYTHON="$(command -v python || true)"
        fi
    fi
    if [[ -z "$WSP_PYTHON" || ! -x "$WSP_PYTHON" ]]; then
        echo "ERROR: Set CONDA_ENV or WSP_PYTHON to a valid Python environment" >&2
        return 1
    fi
    if [[ -n "${CONDA_ENV:-}" ]]; then
        export PATH="$CONDA_ENV/bin:$PATH"
    fi
    export CONDA_ENV WSP_PYTHON
    # The checkout wins over globally installed worldscape_policy packages.
    export PYTHONPATH="$root/src:$root${PYTHONPATH:+:$PYTHONPATH}"

    # Keep compiled-kernel caches on writable node-local storage. Cluster home
    # directories are often read-only inside workers, which disables caching.
    local cache_root="${WSP_CACHE_DIR:-${TMPDIR:-/tmp}/worldscape-policy-${USER:-user}}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$cache_root/torch-extensions}"
    if ! mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"; then
        echo "ERROR: Cannot create training cache directories under $cache_root" >&2
        return 1
    fi

    # Match the optimized legacy training defaults while keeping portable
    # fallbacks when FlashAttention is not installed.
    export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
    if [[ -z "${VLM_ATTN_IMPLEMENTATION:-}" ]]; then
        if "$WSP_PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
            export VLM_ATTN_IMPLEMENTATION=flash_attention_2
        else
            export VLM_ATTN_IMPLEMENTATION=sdpa
        fi
    fi

    if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
    fi
    if [[ -z "${NUM_GPUS:-}" ]]; then
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            local -a visible_gpus
            IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
            NUM_GPUS="${#visible_gpus[@]}"
        elif command -v nvidia-smi >/dev/null 2>&1; then
            NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
            if [[ "$NUM_GPUS" == "0" ]]; then
                NUM_GPUS=1
            fi
        else
            NUM_GPUS=1
        fi
    fi
    if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: NUM_GPUS must be a positive integer, got '$NUM_GPUS'" >&2
        return 2
    fi
    export NUM_GPUS
    export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
    export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
    export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
    export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-29500}"
    if ! [[ "$MLP_WORKER_NUM" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MLP_WORKER_NUM must be a positive integer" >&2
        return 2
    fi

    local world_size=$((MLP_WORKER_NUM * NUM_GPUS))
    export NCCL_TIMEOUT_SECONDS="${NCCL_TIMEOUT_SECONDS:-3600}"
    export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-deepspeed}"
    if (( world_size > 1 )); then
        export DEEPSPEED_MODE="${DEEPSPEED_MODE:-zero2}"
    else
        export DEEPSPEED_MODE="${DEEPSPEED_MODE:-zero2_offload}"
    fi
    case "$DEEPSPEED_MODE" in
        zero2|zero2_offload|zero3) ;;
        *) echo "ERROR: DEEPSPEED_MODE must be zero2, zero2_offload, or zero3" >&2; return 2 ;;
    esac

    export LEARNING_RATE="${LEARNING_RATE:-6e-5}"
    export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    export OPTIMIZER_WARMUP_RATIO="${OPTIMIZER_WARMUP_RATIO:-0.05}"

    export WSP_MODEL_ROOT="${WSP_MODEL_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}/worldscape-policy}"
    export WAN_CKPT_DIR="${WAN_CKPT_DIR:-$WSP_MODEL_ROOT/Wan2.2-TI2V-5B}"
    export CLIP_CKPT_DIR="${CLIP_CKPT_DIR:-$WSP_MODEL_ROOT/Wan2.1-I2V-14B-480P}"
    export TOKENIZER_DIR="${TOKENIZER_DIR:-$WSP_MODEL_ROOT/umt5-xxl}"
    export T5_CKPT_PATH="${T5_CKPT_PATH:-$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth}"
    export VAE_CKPT_PATH="${VAE_CKPT_PATH:-$WAN_CKPT_DIR/Wan2.2_VAE.pth}"
    export CLIP_CKPT_PATH="${CLIP_CKPT_PATH:-$CLIP_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
    export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"
    case "$QWEN_VARIANT" in
        qwen3-2b|2b)
            export Qwen_CKPT_DIR="${Qwen_CKPT_DIR:-$WSP_MODEL_ROOT/Qwen3-VL-2B-Instruct}"
            QWEN_REPO_ID="Qwen/Qwen3-VL-2B-Instruct"
            export VLM_TOKEN_DIM="${VLM_TOKEN_DIM:-2048}"
            ;;
        qwen3-4b|4b)
            export Qwen_CKPT_DIR="${Qwen_CKPT_DIR:-$WSP_MODEL_ROOT/Qwen3-VL-4B-Instruct}"
            QWEN_REPO_ID="Qwen/Qwen3-VL-4B-Instruct"
            export VLM_TOKEN_DIM="${VLM_TOKEN_DIM:-2560}"
            ;;
        *)
            echo "ERROR: QWEN_VARIANT must be qwen3-4b or qwen3-2b, got '$QWEN_VARIANT'" >&2
            return 2
            ;;
    esac
    export WSP_AUTO_DOWNLOAD="${WSP_AUTO_DOWNLOAD:-true}"
    if [[ "$WSP_AUTO_DOWNLOAD" == "true" && ( ! -d "$WAN_CKPT_DIR" || -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ) ]]; then
        wsp_hf_download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN_CKPT_DIR"
    fi
    if [[ "$WSP_AUTO_DOWNLOAD" == "true" && ( ! -d "$TOKENIZER_DIR" || -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ) ]]; then
        wsp_hf_download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
    fi
    if [[ "$WSP_AUTO_DOWNLOAD" == "true" && ! -f "$CLIP_CKPT_PATH" ]]; then
        wsp_hf_download Wan-AI/Wan2.1-I2V-14B-480P \
            models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
            --local-dir "$CLIP_CKPT_DIR"
    fi
    if [[ "$WSP_AUTO_DOWNLOAD" == "true" && "${WSP_MODE:-interactive}" == "auto" && ( ! -d "$Qwen_CKPT_DIR" || -z "$(ls -A "$Qwen_CKPT_DIR" 2>/dev/null)" ) ]]; then
        wsp_hf_download "$QWEN_REPO_ID" --local-dir "$Qwen_CKPT_DIR"
    fi
}

wsp_distributed_train_launch() {
    local root="$1"
    local config="$2"
    shift 2

    wsp_train_setup "$root"

    # Keep explicit custom executables working for smoke tests and integrations.
    if [[ -n "${WSP_TRAIN:-}" ]]; then
        exec "$WSP_TRAIN" --config "$config" "$@"
    fi

    exec "$WSP_PYTHON" -m torch.distributed.run \
        --nnodes "$MLP_WORKER_NUM" \
        --node_rank "$MLP_ROLE_INDEX" \
        --master_addr "$MLP_WORKER_0_HOST" \
        --master_port "$MLP_WORKER_0_PORT" \
        --nproc_per_node "$NUM_GPUS" \
        --module worldscape_policy.cli.train \
        --config "$config" \
        "$@"
}
