"""Restore frozen ICL media from the v2 ICL assets archive; no network or credential access."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARCHIVE_SHA256="d0da120d876abcb328fe95ff4bf2520db181d130efc61a71e15bfe9c2ef9b3a8"
def main():
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);a=p.parse_args()
    if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError("Unexpected ICL assets archive SHA256")
    tasks=json.loads((ROOT/"icl/manifest.json").read_text(encoding="utf-8"))["tasks"]
    count=0
    with zipfile.ZipFile(a.archive) as z:
        for task in tasks:
            base=(ROOT/"icl"/task["task"]).resolve()
            for name,expected in task["files"].items():
                target=(base/name).resolve()
                if base not in target.parents:raise ValueError("Invalid asset path")
                data=z.read("Worldscape_RSI/icl/"+task["task"]+"/"+name)
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError("Invalid asset hash")
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);count+=1
    print("Verified and restored",count,"ICL assets")
if __name__=="__main__":main()
