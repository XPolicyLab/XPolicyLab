"""Download the 14D clean split and prepare a v2 export without re-encoding videos."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

REPO_ID = "OpenMOSS-Team/robotwin2.0-lerobot-v3.0"
REVISION = "4beb760ef21ebfd6f202a8f34109a647a629d8cd"


def digest(path, algorithm="sha256", git_blob=False):
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def download_clean(source, endpoint, workers, proxy=None):
    import httpx
    from huggingface_hub import set_client_factory
    from huggingface_hub.file_download import xet_get
    from huggingface_hub.utils import XetFileData

    # Use the selected proxy consistently in both httpx and the Rust Xet client.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy
    set_client_factory(lambda: httpx.Client(trust_env=False, proxy=proxy, timeout=45))
    source.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["repo_id"], manifest["revision"]) != (REPO_ID, REVISION):
            raise ValueError("Existing source manifest belongs to another dataset/revision.")
    else:
        files = []
        url = f"{endpoint}/api/datasets/{REPO_ID}/tree/{REVISION}/clean?recursive=true&limit=1000"
        with httpx.Client(trust_env=False, proxy=proxy, timeout=45) as client:
            while url:
                response = client.get(url)
                response.raise_for_status()
                files.extend(item for item in response.json() if item["type"] == "file")
                url = response.links.get("next", {}).get("url")
        manifest = {"repo_id": REPO_ID, "revision": REVISION, "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files = manifest["files"]
    print(f"Source: {REPO_ID}@{REVISION}; {len(files)} files, "
          f"{sum(item['size'] for item in files) / 1e9:.2f} GB", flush=True)

    def valid(path, item):
        if not path.is_file() or path.stat().st_size != item["size"]:
            return False
        if item.get("lfs"):
            return digest(path) == item["lfs"]["oid"]
        return digest(path, "sha1", git_blob=True) == item["oid"]

    def download(item):
        target = source / item["path"]
        if valid(target, item):
            return item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        for attempt in range(3):
            try:
                if item.get("xetHash"):
                    xet_get(incomplete_path=partial,
                            xet_file_data=XetFileData(file_hash=item["xetHash"],
                                refresh_route=f"{endpoint}/api/datasets/{REPO_ID}/xet-read-token/{REVISION}"),
                            headers={}, expected_size=item["size"], displayed_filename=item["path"])
                else:
                    with httpx.Client(trust_env=False, proxy=proxy, timeout=45, follow_redirects=True) as client:
                        response = client.get(f"{endpoint}/datasets/{REPO_ID}/resolve/{REVISION}/{item['path']}")
                        response.raise_for_status()
                        partial.write_bytes(response.content)
                if not valid(partial, item):
                    partial.unlink(missing_ok=True)
                    raise IOError(f"Size or checksum mismatch for {item['path']}")
                partial.replace(target)
                return item["path"]
            except Exception as exc:
                print(f"Retry {attempt + 1}: {item['path']}: {type(exc).__name__}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, item) for item in files]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                name = future.result()
                if count % 5 == 0 or "/meta/" in name or count == len(files):
                    print(f"Verified {count}/{len(files)}: {name}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    (source / "download_verified.json").write_text(json.dumps({
        "repo_id": REPO_ID, "revision": REVISION, "verified_files": len(files),
        "verified_bytes": sum(item["size"] for item in files)}, indent=2) + "\n")


def prepare_export(source, destination):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    info = json.loads((source / "meta/info.json").read_text())
    for feature in ("action", "observation.state"):
        if info["features"][feature]["shape"] != [14]:
            raise ValueError(f"Expected 14D joint data; {feature} is {info['features'][feature]['shape']}.")
    provenance = {"repo_id": REPO_ID, "revision": REVISION, "source": str(source.resolve())}
    marker = destination / "source.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise ValueError(f"{destination} belongs to another source; choose a new destination.")
    if (destination / "meta/info.json").exists() and not marker.exists():
        raise ValueError(f"Refusing to overwrite an existing unrelated dataset at {destination}.")
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(provenance, indent=2) + "\n")
    metadata = destination / "meta"
    metadata.mkdir(exist_ok=True)
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pandas().reset_index()
    text_column = "task" if "task" in tasks.columns else next(name for name in tasks.columns if name != "task_index")
    task_rows = [{"task_index": int(row["task_index"]), "task": str(row[text_column])} for row in tasks.to_dict("records")]
    episodes = []
    for path in sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in episodes] != list(range(int(info["total_episodes"]))):
        raise ValueError("Source episode metadata is incomplete or duplicated.")
    cameras = [name for name, field in info["features"].items() if field["dtype"] == "video"]

    @lru_cache(maxsize=2)
    def data_file(chunk, index):
        return pq.read_table(source / info["data_path"].format(chunk_index=chunk, file_index=index))

    exported = []
    used_tasks = set()
    for row in episodes:
        index = int(row["episode_index"])
        chunk = index // 1000
        target = destination / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
        length = int(row["length"])
        if not target.is_file() or pq.read_metadata(target).num_rows != length:
            table = data_file(int(row["data/chunk_index"]), int(row["data/file_index"]))
            episode = table.filter(pc.equal(table["episode_index"], index)).replace_schema_metadata(None)
            if episode.num_rows != length:
                raise ValueError(f"Episode {index}: expected {length} rows, got {episode.num_rows}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".partial")
            pq.write_table(episode, partial)
            partial.replace(target)
        if "task_index" in pq.read_schema(target).names:
            used_tasks.update(pq.read_table(target, columns=["task_index"])["task_index"].unique().to_pylist())
        offsets = {}
        for camera in cameras:
            prefix = f"videos/{camera}"
            video = source / info["video_path"].format(video_key=camera,
                chunk_index=int(row[prefix + "/chunk_index"]), file_index=int(row[prefix + "/file_index"]))
            if not video.is_file():
                raise FileNotFoundError(video)
            link = destination / f"videos/chunk-{chunk:03d}/{camera}/episode_{index:06d}.mp4"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != video.resolve():
                    raise ValueError(f"Unexpected existing video link: {link}")
            elif link.exists():
                raise ValueError(f"Refusing to replace an existing video: {link}")
            else:
                link.symlink_to(os.path.relpath(video.resolve(), link.parent.resolve()))
            offsets[camera] = float(row[prefix + "/from_timestamp"])
        exported.append({"episode_index": index, "length": length, "tasks": list(row["tasks"]),
                         "video_timestamp_offsets": offsets})
        if (index + 1) % 250 == 0:
            print(f"Prepared {index + 1}/{len(episodes)} episodes", flush=True)
    if sum(row["length"] for row in exported) != int(info["total_frames"]):
        raise ValueError("Exported frame count does not match the source.")
    (metadata / "episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported))
    task_rows = [row for row in task_rows if not used_tasks or row["task_index"] in used_tasks]
    (metadata / "tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_rows))
    (metadata / "stats.json").write_bytes((source / "meta/stats.json").read_bytes())
    output_info = dict(info, codebase_version="v2.0", chunks_size=1000, total_tasks=len(task_rows),
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    for camera in cameras:
        feature = dict(output_info["features"][camera])
        feature["shape"] = [3, int(feature["info"]["video.height"]), int(feature["info"]["video.width"])]
        feature["names"] = ["channels", "height", "width"]
        output_info["features"][camera] = feature
    (metadata / "info.json").write_text(json.dumps(output_info, indent=2) + "\n")
    print(f"Prepared {len(exported)} episodes / {info['total_frames']} frames at {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Download directory for the LeRobot v3.0 source (its clean/ split lands here).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination of the LeRobot v2.0 training export.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy", default=(os.environ.get("WAM_DOWNLOAD_PROXY")
                        or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if not args.skip_download:
        download_clean(args.source, args.endpoint.rstrip("/"), args.workers, args.proxy)
    prepare_export(args.source / "clean", args.output)


if __name__ == "__main__":
    main()
