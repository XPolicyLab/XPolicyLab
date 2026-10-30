"""Check hashes by default; explicitly refresh a NEW INACTIVE candidate after edits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value): p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-new-release",action="store_true",
                        help="Inactive candidate only; does not validate old evidence against new code")
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("release_tools",ROOT/"harness/release.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    catalog_path=ROOT/"harness/configs/wam_skills.json"
    catalog=json.loads(catalog_path.read_text())
    for value in catalog.values():
        for name,expected in list(value["source_files"].items()):
            actual=digest(module.contained(ROOT,name))
            if args.write_new_release:value["source_files"][name]=actual
            elif actual!=expected:raise RuntimeError("Source binding changed: "+name)
    target=ROOT/"harness/release.json";old=target.read_bytes()
    previous=hashlib.sha256(old).hexdigest()
    if args.write_new_release:
        write(catalog_path,catalog)
        history=ROOT/"validation/release-history";history.mkdir(parents=True,exist_ok=True)
        backup=history/(previous+".json")
        if backup.exists() and backup.read_bytes()!=old:raise RuntimeError("Conflicting history")
        if not backup.exists():backup.write_bytes(old)
        target.unlink()
        try:module.freeze(ROOT/"harness")
        except BaseException:
            if target.exists():target.unlink()
            target.write_bytes(old)
            raise
    release=module.FrozenRelease(target)
    if args.write_new_release:
        path=ROOT/"MANIFEST.json";manifest=json.loads(path.read_text())
        manifest.update(active_release_sha256=release.sha256,revision="generic-remediation-20261024")
        write(path,manifest)
        path=ROOT/"VALIDATION.json";validation=json.loads(path.read_text())
        if validation.get("release_sha256")!=release.sha256:
            validation.setdefault("superseded_checks",[]).append({
                "release_sha256":validation.get("release_sha256"),
                "checks":validation.get("current_release_checks",{}),
                "reason":"Source/documentation changed; earlier checks remain historical"})
            validation.update(release_sha256=release.sha256,
                current_release_checks={"status":"requires_validation_for_new_identity"})
            write(path,validation)
    manifest=json.loads((ROOT/"MANIFEST.json").read_text())
    if manifest.get("active_release_sha256")!=release.sha256:
        raise RuntimeError("MANIFEST mismatch; refresh the inactive candidate")
    print(json.dumps({"release_sha256":release.sha256,"source_hashes_verified":True,
                      "model_or_robot_validation":False}))
if __name__=="__main__":main()
