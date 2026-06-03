# drop_cam_fields_to_data1.py
from __future__ import annotations
import json
import shutil
from pathlib import Path
import pyarrow.parquet as pq
LEROBOT_ROOT = Path("/mnt/workspace/yangyandan/download/lerobot")
DATA_DIR = LEROBOT_ROOT / "data_save"
OUT_DIR = LEROBOT_ROOT / "data"
FIELDS_TO_DROP = [
    "observation.images.cam_high",
    "observation.images.cam_low",
]
# 是否覆盖已存在的 data1
OVERWRITE_OUT_DIR = False
# 是否同步更新 meta/info.json 里的 features 与 data_path（避免加载器仍指向旧字段/旧data目录）
UPDATE_META_INFO = False
def drop_fields_and_save(lerobot_root: Path) -> None:
    data_dir = lerobot_root / "data_save"
    out_dir = lerobot_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data目录不存在: {data_dir}")
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"在 {data_dir} 下没有找到 parquet 文件")
    if out_dir.exists():
        if not OVERWRITE_OUT_DIR:
            # 只要里面已有 parquet 就直接拒绝，避免误覆盖
            if any(out_dir.rglob("*.parquet")):
                raise RuntimeError(f"{out_dir} 已存在且包含 parquet，请先手动处理或将 OVERWRITE_OUT_DIR=True")
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_set = set(FIELDS_TO_DROP)
    for in_path in parquet_files:
        rel = in_path.relative_to(data_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(in_path)
        # Parquet 物理 schema（pf.schema）会把 fixed_size_list 拆成子列，名字常是 element；
        # 逻辑列名用 Arrow schema：observation.state / action 等顶层字段。
        col_names = list(pf.schema_arrow.names)
        keep_cols = [c for c in col_names if c not in drop_set]
        # 如果原文件里没这些字段，就仍然原样拷贝（只是不读取被drop的列）
        table = pf.read(columns=keep_cols) if keep_cols != col_names else pf.read()
        pq.write_table(table, out_path)
    if UPDATE_META_INFO:
        info_path = lerobot_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"meta/info.json 不存在: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        for k in FIELDS_TO_DROP:
            features.pop(k, None)
        info["features"] = features
        data_path = info.get("data_path", "")
        if isinstance(data_path, str) and data_path.startswith("data/"):
            info["data_path"] = data_path.replace("data/", "data1/", 1)
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("完成：已将字段删除并保存到 data1")
if __name__ == "__main__":
    drop_fields_and_save(LEROBOT_ROOT)
