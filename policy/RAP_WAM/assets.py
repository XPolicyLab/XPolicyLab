"""Validate the exact retained model and its portable inference bundle."""
import hashlib
import json
from pathlib import Path
import torch

SOURCE_SHA256 = "4d0f34bb7cd6f3520dbd4742780b410bae4a39c83b2fb99aa1798dd13c96e3d9"
TASK_CONTEXT_SHA256 = "845c145ddad653c8566f5dbfb909aaf9a074a5f698c6c3aaf23b51423cc5d569"
MODEL_FINGERPRINT = "c3d500da9fcec1e83aa4414f71f68b329bddb0a5553fd0addfaf28dcf76f37ea"
REQUIRED = {"model.pt", "inference_config.json", "task_context.pt", "task_strings.json", "model_tensors.json"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_manifest(state):
    result = {}
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor model entry: {name}")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest = hashlib.sha256()
        view = memoryview(raw)
        for offset in range(0, len(view), 16 << 20):
            digest.update(view[offset:offset + (16 << 20)])
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest.hexdigest()}
    return result


def manifest_fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_bundle(root, verify_files=True):
    root = Path(root).expanduser().resolve()
    config = json.loads((root / "inference_config.json").read_text())
    if config.get("format") != "rap-wam-v58-light-inference-v1":
        raise ValueError("Unsupported inference bundle format")
    if config.get("source_checkpoint_sha256") != SOURCE_SHA256 or config.get("model_fingerprint") != MODEL_FINGERPRINT:
        raise ValueError("Expected the fixed historical s742500 model")
    checksums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or name != Path(name).name or name in checksums:
            raise ValueError("Invalid checksum entry")
        checksums[name] = digest
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_files and sha256_file(path) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    if not REQUIRED.issubset(checksums):
        raise ValueError("Incomplete inference bundle")
    manifest = json.loads((root / "model_tensors.json").read_text())
    if len(manifest) != 886 or manifest_fingerprint(manifest) != MODEL_FINGERPRINT:
        raise ValueError("Wrong model tensor manifest")
    if sha256_file(root / "task_context.pt") != TASK_CONTEXT_SHA256:
        raise ValueError("Task context is not the original trained table")
    table = torch.load(root / "task_context.pt", map_location="cpu", weights_only=True)
    tasks = json.loads((root / "task_strings.json").read_text())
    if not isinstance(tasks, list) or len(tasks) != 35 or len(set(tasks)) != 35 or tasks != table.get("tasks"):
        raise ValueError("Task vocabulary differs from the trained context table")
    embeddings = table.get("embedding_table")
    if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != (35, 512, 4096) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid trained language-context table")
    return config, tasks
