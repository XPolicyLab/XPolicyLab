from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    local_dir="./Qwen3-VL-2B-Instruct",
    repo_type="model",
    resume_download=True,
)

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir="./Qwen3-VL-8B-Instruct",
    repo_type="model",
    resume_download=True,
)
