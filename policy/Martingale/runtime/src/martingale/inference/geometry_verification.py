"""Read-only integrity and completeness checks for frozen RGB geometry evidence.

Verification uses the saved report, original RGB captures and camera metadata.
It checks no expected geometry values. It does not execute agent derivations or
certify physical accuracy, correspondence identity, or impossibility proofs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

from .geometry_fitting_specs import canonical_task


REPORT_SCHEMA = "martingale-rgb-geometry-v1"
VERIFICATION_SCHEMA = "martingale-rgb-geometry-verification-v1"
COMPLETION_POLICY = "all_estimated_or_evidence_backed_impossible_v1"
ALLOWED_INPUTS = frozenset({"RGB", "camera_calibration", "robot_proprioception"})
UNITS = frozenset({"m", "m2", "rad", "deg", "px", "count", "dimensionless", "mixed"})
IMPOSSIBILITY_CATEGORIES = frozenset({"non_identifiable", "observation_inaccessible", "outside_permitted_inputs"})


def _load(path):
    raw = Path(path).read_bytes()

    def nonfinite(value):
        raise ValueError("Non-finite JSON number: " + value)

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(raw, parse_constant=nonfinite, object_pairs_hook=unique_pairs), hashlib.sha256(raw).hexdigest()


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _flatten(value, path=""):
    """Validate a nonempty finite numeric tree without assigning physical truth."""
    if _number(value):
        return {path: float(value)}
    if isinstance(value, dict) and value and all(isinstance(k, str) and k for k in value):
        result = {}
        for key, child in value.items():
            part = key.replace("~", "~0").replace("/", "~1")
            result.update(_flatten(child, path + "/" + part))
        return result
    if isinstance(value, list) and value:
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, path + "/" + str(index)))
        return result
    raise ValueError("Expected a nonempty finite numeric tree at " + (path or "/"))


def _shape(value):
    if isinstance(value, dict):
        return ("mapping", {k: _shape(v) for k, v in value.items()})
    if isinstance(value, list):
        return ("array", [_shape(v) for v in value])
    return "number"


def _text(value, label, minimum=10, maximum=20000):
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ValueError("Missing or invalid " + label)


def _matrix(value, size, label):
    if (not isinstance(value, list) or len(value) != size
            or any(not isinstance(row, list) or len(row) != size for row in value)
            or any(not _number(x) for row in value for x in row)):
        raise ValueError("Invalid " + label)


def _validate_calibration(calibration):
    if not isinstance(calibration, dict) or set(calibration) != {"shape", "intrinsic_matrix", "extrinsic_matrix"}:
        raise ValueError("Only RGB shape, camera intrinsics and camera-to-world pose are allowed")
    shape = calibration["shape"]
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
            or any(type(x) is not int or x <= 0 for x in shape)):
        raise ValueError("Invalid original RGB shape")
    k, pose = calibration["intrinsic_matrix"], calibration["extrinsic_matrix"]
    _matrix(k, 3, "camera intrinsics")
    _matrix(pose, 4, "camera pose")
    if (min(k[0][0], k[1][1]) <= 0
            or any(abs(x - y) > 1e-8 for x, y in zip(k[2], [0, 0, 1]))
            or abs(k[0][1]) > 1e-8 or abs(k[1][0]) > 1e-8):
        raise ValueError("Expected rectified pinhole camera intrinsics")
    if any(abs(x - y) > 1e-8 for x, y in zip(pose[3], [0, 0, 0, 1])):
        raise ValueError("Invalid homogeneous camera pose")
    r = [row[:3] for row in pose[:3]]
    for a in range(3):
        for b in range(3):
            if abs(sum(r[i][a] * r[i][b] for i in range(3)) - int(a == b)) > 2e-5:
                raise ValueError("Camera pose must be rigid")
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if abs(det - 1) > 2e-5:
        raise ValueError("Camera rotation must preserve handedness")


def _evidence_audit(report, workspace):
    """Check capture bytes and allowlisted metadata, not semantic correctness."""
    views = report.get("views")
    if not isinstance(views, dict) or not views:
        raise ValueError("Frozen report must retain RGB views")
    workspace = Path(workspace).resolve()
    evidence_root = workspace / "fit_evidence"
    verified, problems = [], {}
    fields = {"camera", "step", "calibration", "image"}
    legacy_fields = fields | {"calibration_sha256", "rgb_sha256", "png_sha256"}
    for ident, view in views.items():
        try:
            _text(ident, "view ID", 1, 128)
            if not isinstance(view, dict) or set(view) not in (fields, legacy_fields):
                raise ValueError("View metadata must contain only the capture fields")
            _text(view["camera"], "camera identifier", 1, 128)
            if type(view["step"]) is not int or view["step"] < 0:
                raise ValueError("Invalid capture step")
            relative = Path(view["image"])
            image_path = (workspace / relative).resolve()
            if relative.is_absolute() or not image_path.is_relative_to(evidence_root):
                raise ValueError("Evidence image must remain inside workspace/fit_evidence")
            raw = image_path.read_bytes()
            if "png_sha256" in view and hashlib.sha256(raw).hexdigest() != view["png_sha256"]:
                raise ValueError("Retained PNG checksum mismatch")
            metadata, _ = _load(image_path.with_suffix(".json"))
            if metadata != view:
                raise ValueError("Retained view metadata differs from frozen report")
            calibration = view["calibration"]
            _validate_calibration(calibration)
            digest = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
            if "calibration_sha256" in view and digest != view["calibration_sha256"]:
                raise ValueError("Camera calibration checksum mismatch")
            with Image.open(io.BytesIO(raw)) as rgb:
                if rgb.format != "PNG" or rgb.mode != "RGB":
                    raise ValueError("Saved capture must be an original RGB PNG")
                if list(calibration["shape"]) != [rgb.height, rgb.width, 3]:
                    raise ValueError("Camera shape differs from original RGB")
                rgb.load()
                if "rgb_sha256" in view and hashlib.sha256(rgb.tobytes()).hexdigest() != view["rgb_sha256"]:
                    raise ValueError("Original RGB checksum mismatch")
            verified.append(ident)
        except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as exc:
            problems[ident] = str(exc)
    return {"verified_views": verified, "view_errors": problems,
            "scope": "Saved RGB bytes, capture metadata and calibration integrity; not an attestation of capture origin or correspondence identity."}


def _landmark_views(landmark, report, verified):
    if not isinstance(landmark, dict) or landmark.get("accepted") is not True:
        raise ValueError("Evidence must cite an accepted RGB landmark")
    xyz = landmark.get("xyz")
    if not isinstance(xyz, list) or len(xyz) != 3 or any(not _number(x) for x in xyz):
        raise ValueError("Invalid landmark coordinates")
    observations = landmark.get("observations")
    if not isinstance(observations, list) or not 2 <= len(observations) <= 32:
        raise ValueError("Landmark needs two to 32 observations")
    ids = []
    for ob in observations:
        if not isinstance(ob, dict) or set(ob) - {"view_id", "uv", "sigma_px"}:
            raise ValueError("Only RGB pixel observations may support landmarks")
        ident, uv, sigma = ob.get("view_id"), ob.get("uv"), ob.get("sigma_px", 1.)
        if not isinstance(ident, str) or ident not in verified or ident in ids:
            raise ValueError("Landmark needs distinct verified views")
        h, w, _ = report["views"][ident]["calibration"]["shape"]
        if (not isinstance(uv, list) or len(uv) != 2 or any(not _number(x) for x in uv)
                or not 0 <= uv[0] < w or not 0 <= uv[1] < h):
            raise ValueError("Landmark pixels must lie inside original RGB")
        if not _number(sigma) or not .25 <= sigma <= 20:
            raise ValueError("Invalid landmark pixel uncertainty")
        ids.append(ident)
    return ids


def _validate_evidence(record, report, verified):
    views = record.get("view_ids")
    if (not isinstance(views, list) or not views or any(not isinstance(v, str) for v in views)
            or len(set(views)) != len(views) or any(v not in verified for v in views)):
        raise ValueError("Record must cite distinct, verified retained view IDs")
    evidence = record.get("evidence")
    if (not isinstance(evidence, list) or not 1 <= len(evidence) <= 256
            or any(not isinstance(ref, str) for ref in evidence)):
        raise ValueError("Missing or invalid RGB evidence references")
    supported = set()
    for ref in evidence:
        if ref in verified:
            supported.add(ref)
        else:
            if ref not in report["landmarks"]:
                raise ValueError("Unknown RGB evidence: " + ref)
            supported.update(_landmark_views(report["landmarks"][ref], report, verified))
    if supported != set(views):
        raise ValueError("Evidence references do not match the record's view IDs")
    return views


def _validate_estimate(estimate, report, verified):
    if estimate.get("source") != "rgb_estimate":
        raise ValueError("Missing RGB estimate provenance")
    if estimate.get("unit") not in UNITS:
        raise ValueError("Unsupported measurement unit")
    values = _flatten(estimate.get("value"))
    uncertainty = _flatten(estimate.get("uncertainty"))
    if min(uncertainty.values()) < 0:
        raise ValueError("Uncertainty must be nonnegative")
    if not _number(estimate["uncertainty"]) and _shape(estimate["uncertainty"]) != _shape(estimate["value"]):
        raise ValueError("Uncertainty must be scalar or match the value structure")
    for field in ("method", "derivation"):
        _text(estimate.get(field), "measurement " + field)
    views = _validate_evidence(estimate, report, verified)
    if estimate["unit"] in {"m", "m2", "rad", "deg", "mixed"}:
        centers = [[report["views"][ident]["calibration"]["extrinsic_matrix"][row][3] for row in range(3)] for ident in views]
        if len(centers) < 2 or max(math.dist(a, b) for a in centers for b in centers) < .01:
            raise ValueError("Metric estimate lacks distinct camera centers")
    return len(values)


def _validate_impossible(claim, report, verified):
    if claim.get("source") != "agent_rgb_impossibility_claim" or claim.get("review_required") is not True:
        raise ValueError("Impossibility must be an agent claim requiring review")
    if claim.get("category") not in IMPOSSIBILITY_CATEGORIES:
        raise ValueError("Unsupported impossibility category; resource exhaustion is incomplete")
    _text(claim.get("reason"), "impossibility reason")
    _text(claim.get("justification"), "impossibility justification", 40)
    for label, fields, minimum in (("attempts", {"method", "result"}, 2),
                                    ("alternatives", {"approach", "blocked_by"}, 1)):
        records = claim.get(label)
        if not isinstance(records, list) or not minimum <= len(records) <= 100:
            raise ValueError("Missing or invalid impossibility " + label)
        for record in records:
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError("Invalid impossibility " + label + " fields")
            for field in fields:
                _text(record[field], label + " " + field)
    if len({r["method"].strip().casefold() for r in claim["attempts"]}) != len(claim["attempts"]):
        raise ValueError("Impossibility must document distinct attempted methods")
    _validate_evidence(claim, report, verified)


def verify_report(report_path, *, workspace=None):
    """Audit frozen evidence and completion without reading target geometry values.

    An integrity pass can include impossibility claims, always flagged for review.
    It is not a claim that every parameter was measured or that any estimate or
    impossibility argument is physically correct. Malformed headers raise ValueError;
    invalid evidence, estimates, claims or incomplete coverage produce a failed audit.
    """
    report_path = Path(report_path)
    report, report_sha = _load(report_path)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise ValueError("Only a complete, frozen RGB geometry report may be verified")
    if report.get("reference_values_visible") is not False:
        raise ValueError("Report does not declare isolation from reference values")
    inputs = report.get("inputs")
    if (not isinstance(inputs, list) or any(not isinstance(i, str) for i in inputs)
            or "RGB" not in inputs or len(set(inputs)) != len(inputs) or set(inputs) - ALLOWED_INPUTS):
        raise ValueError("Report inputs must be RGB with only allowed camera calibration/proprioception")
    _text(report.get("task"), "report task", 1, 128)
    requirements, estimates = report.get("requirements"), report.get("estimates")
    if not isinstance(requirements, dict) or not isinstance(estimates, dict) or not isinstance(report.get("landmarks"), dict):
        raise ValueError("Report requirements, estimates and landmarks must be mappings")
    if report.get("completion_policy") != COMPLETION_POLICY:
        raise ValueError("Missing or unsupported completion policy")
    for name, description in requirements.items():
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,127}", name):
            raise ValueError("Invalid parameter name")
        _text(description, "requirement description", 1, 2000)
    evidence = _evidence_audit(report, workspace or report_path.parent / "workspace")
    verified = set(evidence["verified_views"])
    rows = {}
    for name in sorted(set(requirements) | set(estimates)):
        record = estimates.get(name)
        row = {"status": "missing", "valid": False}
        if name not in requirements:
            row.update(status="undeclared", reason="Estimate has no declared requirement")
        elif not isinstance(record, dict):
            if name in estimates:
                row.update(status="invalid_record", reason="Parameter record must be an object")
        elif record.get("status") == "unresolved":
            row.update(status="unresolved", reason=record.get("reason", "Still under investigation"))
        elif record.get("status") in {"estimated", "impossible"}:
            try:
                if record["status"] == "estimated":
                    leaves = _validate_estimate(record, report, verified)
                    row.update(status="estimated", valid=True, numeric_components=leaves, unit=record["unit"])
                else:
                    _validate_impossible(record, report, verified)
                    row.update(status="impossible_claim", valid=True, category=record["category"],
                               review_required=True, proof_verified=False)
                row["view_ids"] = record["view_ids"]
            except (ValueError, KeyError, TypeError) as exc:
                row.update(status="invalid_" + record["status"], reason=str(exc))
        else:
            row.update(status="invalid_record", reason="Unrecognized parameter status")
        rows[name] = row
    errors = []
    progress = report.get("progress")
    expected_progress = {
        "requested": len(requirements),
        "estimated": sum(isinstance(estimates.get(n), dict) and estimates[n].get("status") == "estimated" for n in requirements),
        "impossible": sum(isinstance(estimates.get(n), dict) and estimates[n].get("status") == "impossible" for n in requirements),
        "remaining": sorted(n for n in requirements if not isinstance(estimates.get(n), dict) or estimates[n].get("status") not in {"estimated", "impossible"}),
        "unresolved": sorted(n for n in requirements if isinstance(estimates.get(n), dict) and estimates[n].get("status") == "unresolved"),
        "completion_policy": COMPLETION_POLICY,
    }
    expected_progress["ready_to_finish"] = not expected_progress["remaining"] and bool(report["views"])
    if not isinstance(progress, dict) or progress != expected_progress:
        errors.append("Saved progress does not match declared requirements and parameter records")
    failed = [name for name, row in rows.items() if not row["valid"]]
    review = [name for name, row in rows.items() if row.get("review_required")]
    valid = not failed and not errors and not evidence["view_errors"]
    return {
        "schema": VERIFICATION_SCHEMA, "task": canonical_task(report["task"]),
        "status": "pass" if valid else "fail", "valid": valid,
        "report_sha256": report_sha, "report_path": str(report_path), "report_frozen": True,
        "evidence_audit": evidence, "errors": errors, "failed_parameters": failed,
        "review_required": bool(review), "review_parameters": review,
        "completeness": {"requested": len(requirements),
                         "estimated": sum(row["status"] == "estimated" for row in rows.values()),
                         "impossible_claims": len(review), "invalid_or_outstanding": len(failed)},
        "parameters": rows,
        "limitations": "Checks validate retained RGB evidence integrity, documented derivations and completion structure. "
                       "Impossibility claims require review.",
    }


def write_verification(report_path, output_path, *, workspace=None):
    """Atomically write a separate audit without mutating the frozen fit."""
    output_path = Path(output_path)
    if output_path.resolve() == Path(report_path).resolve():
        raise ValueError("Verification output must not overwrite the frozen report")
    result = verify_report(report_path, workspace=workspace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    try:
        result = write_verification(args.report, args.output, workspace=args.workspace)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps({"status": result["status"], "task": result["task"], "completeness": result["completeness"],
                      "review_required": result["review_required"], "review_parameters": result["review_parameters"],
                      "report_sha256": result["report_sha256"], "output": str(args.output)}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
