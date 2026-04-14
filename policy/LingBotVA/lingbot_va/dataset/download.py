import tarfile
from pathlib import Path

import gdown

file_id = "1D52nK4ZOJmWBXKv1nWrLb9YBwq8nKa_b"
archive_path = Path("pick-n-place-sq-lerobot-v21.tgz")
extract_dir = Path("pick-n-place-sq-lerobot-v21")

url = f"https://drive.google.com/uc?id={file_id}"

print("Downloading...")
gdown.download(url, str(archive_path), quiet=False)

extract_dir.mkdir(parents=True, exist_ok=True)

print("Extracting...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print(f"Done. Extracted to: {extract_dir.resolve()}")

