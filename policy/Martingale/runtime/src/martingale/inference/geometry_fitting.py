"""RGB evidence ledger and task-independent calibrated multi-view fitting.

Camera calibration and robot proprioception are allowed metric sensor inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import copy
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .control import atomic_json
from .geometry_fitting_specs import canonical_task, requirements
from .observation_views import observation_bundle


COMPLETION_POLICY = "all_estimated_or_evidence_backed_impossible_v1"
IMPOSSIBILITY_CATEGORIES = frozenset({"non_identifiable", "observation_inaccessible", "outside_permitted_inputs"})


def camera_arrays(calibration):
    """Read only explicitly allowed camera fields; camera looks along local -z."""
    k = np.asarray(calibration["intrinsic_matrix"], dtype=float)
    pose = np.asarray(calibration["extrinsic_matrix"], dtype=float)
    if k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("Expected 3x3 intrinsics and 4x4 camera-to-world pose")
    if not np.isfinite(k).all() or not np.isfinite(pose).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid camera calibration")
    if not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0):
        raise ValueError("Expected rectified pinhole camera intrinsics")
    if (not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-5)):
        raise ValueError("Camera pose must be rigid")
    return k, pose


def project(calibration, xyz):
    k, pose = camera_arrays(calibration)
    local = (np.asarray(xyz, float) - pose[:3, 3]) @ pose[:3, :3]
    depth = -local[..., 2]
    uv = np.stack([k[0, 2] + k[0, 0] * local[..., 0] / depth,
                   k[1, 2] - k[1, 1] * local[..., 1] / depth], axis=-1)
    return uv, depth


def triangulate(views, observations):
    """Fit one physical landmark; reject weak parallax and inconsistent matches.

    observations=[{view_id, uv: [column,row], sigma_px: positive float}, ...].
    sigma_px defaults to one pixel. Covariance is conditional on camera calibration
    and correspondence identity; it does not include shared calibration error.
    """
    if not isinstance(observations, list) or not 2 <= len(observations) <= 32:
        raise ValueError("Use two to 32 views of the SAME physical landmark")
    origins, rays, pixels, sigmas, calibrations, seen = [], [], [], [], [], set()
    for ob in observations:
        if set(ob) - {"view_id", "uv", "sigma_px"}:
            raise ValueError("Only view_id, uv and sigma_px are accepted")
        ident = ob["view_id"]
        if ident in seen:
            raise ValueError("Duplicate view cannot supply independent evidence")
        seen.add(ident)
        view = views[ident]
        uv = np.asarray(ob["uv"], float)
        sigma = float(ob.get("sigma_px", 1.0))
        h, w = view["calibration"]["shape"][:2]
        if uv.shape != (2,) or not np.isfinite(uv).all() or not (0 <= uv[0] < w and 0 <= uv[1] < h):
            raise ValueError("Pixels must be [column,row] inside the original RGB image")
        if not math.isfinite(sigma) or sigma < 0.25 or sigma > 20:
            raise ValueError("sigma_px must be between 0.25 and 20 pixels")
        k, pose = camera_arrays(view["calibration"])
        ray = pose[:3, :3] @ np.array([(uv[0] - k[0, 2]) / k[0, 0], -(uv[1] - k[1, 2]) / k[1, 1], -1.0])
        origins.append(pose[:3, 3]); rays.append(ray / np.linalg.norm(ray))
        pixels.append(uv); sigmas.append(sigma); calibrations.append(view["calibration"])
    origins, rays = np.asarray(origins), np.asarray(rays)
    baseline = float(np.linalg.norm(origins[:, None] - origins[None, :], axis=-1).max())
    angle = float(np.degrees(np.arccos(np.clip(np.abs(rays @ rays.T), 0, 1))).max())
    if baseline < 0.01 or angle < 1.0:
        return {"accepted": False, "reason": "insufficient_parallax", "baseline_m": baseline, "parallax_deg": angle}
    projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
    normal = projectors.sum(axis=0)
    if np.linalg.cond(normal) > 1e8:
        return {"accepted": False, "reason": "ill_conditioned_rays"}
    initial = np.linalg.solve(normal, np.einsum("nij,nj->i", projectors, origins))

    def residual(xyz):
        return np.concatenate([(project(c, xyz)[0] - px) / sigma
                               for c, px, sigma in zip(calibrations, pixels, sigmas)])

    fit = least_squares(residual, initial, loss="linear", max_nfev=100)
    errors = np.array([np.linalg.norm(project(c, fit.x)[0] - px) for c, px in zip(calibrations, pixels)])
    front = all(project(c, fit.x)[1] > 0 for c in calibrations)
    accepted = bool(fit.success and front and np.isfinite(fit.x).all()
                    and (errors <= np.minimum(4.0, 3 * np.asarray(sigmas))).all()
                    and np.linalg.cond(fit.jac.T @ fit.jac) < 1e8)
    result = {"accepted": accepted, "xyz": fit.x.tolist(), "reprojection_px": errors.tolist(),
              "rms_px": float(np.sqrt(np.mean(errors ** 2))), "baseline_m": baseline, "parallax_deg": angle,
              "observations": observations, "uncertainty_scope": "pixel noise conditional on correct identity and camera calibration"}
    if accepted:
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * max(1.0, float(np.dot(fit.fun, fit.fun)) / (2 * len(observations) - 3))
        result.update(covariance_m2=covariance.tolist(), sigma_xyz_m=np.sqrt(np.diag(covariance)).tolist())
    else:
        result["reason"] = "inconsistent_correspondences_or_behind_camera"
    return result


def numeric(value, *, nonnegative=False):
    """JSON numeric trees, preserving per-instance names and dimensions."""
    if isinstance(value, dict) and value:
        return {str(k): numeric(v, nonnegative=nonnegative) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError("Empty numeric values are not measurements")
        return [numeric(v, nonnegative=nonnegative) for v in value]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Expected a finite numeric value, array or mapping")
    out = float(value)
    if not math.isfinite(out) or (nonnegative and out < 0):
        raise ValueError("Invalid numeric value/uncertainty")
    return out


def same_structure(left, right):
    """Compare normalized numeric trees without confusing arrays and mappings."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_structure(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_structure(a, b) for a, b in zip(left, right))
    return True


class GeometryFitting:
    """One episode's append-only evidence and explicitly frozen calibration report."""

    def __init__(self, task, workspace):
        self.task = canonical_task(task)
        self.workspace = Path(workspace)
        self.root = self.workspace / "fit_evidence"
        self.root.mkdir(parents=True, exist_ok=True)
        self.requested = requirements(task)
        self.views, self.landmarks, self.estimates = {}, {}, {}
        self.finished = False
        self.report_path = self.workspace.parent / "geometry_fit.json"
        self._save()

    def _mutable(self):
        if self.finished:
            raise ValueError("Calibration report is frozen")

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,127}", name):
            raise ValueError("Use a short identifier for the parameter/landmark")
        return name

    def report(self):
        return copy.deepcopy({"schema": "martingale-rgb-geometry-v1", "task": self.task,
                "status": "complete" if self.finished else "collecting",
                "inputs": ["RGB", "camera_calibration", "robot_proprioception"],
                "reference_values_visible": False, "requirements": self.requested,
                "completion_policy": COMPLETION_POLICY, "progress": self.fit_progress(),
                "views": self.views, "landmarks": self.landmarks, "estimates": self.estimates,
                "limitations": "Uncertainty is conditional on robot/camera calibration; control choices need separate experiments. "
                               "Impossibility claims require review."})

    def _save(self):
        atomic_json(self.report_path if self.finished else self.workspace.parent / "geometry_fit.partial.json", self.report())

    def fit_requirements(self):
        """Return requested parameter names/descriptions, never reference values."""
        return dict(self.requested)

    def fit_progress(self):
        """Return outstanding parameters and completion counts, without reference values.

        Missing parameters and fit_unresolved notes remain outstanding. Finishing
        requires an estimate or an evidence-backed fit_impossible claim for each
        parameter. Counts validate report coverage, not the truth of its claims.
        """
        statuses = {name: self.estimates.get(name, {}).get("status") for name in self.requested}
        remaining = sorted(name for name, status in statuses.items() if status not in {"estimated", "impossible"})
        return {"requested": len(statuses), "estimated": sum(s == "estimated" for s in statuses.values()),
                "impossible": sum(s == "impossible" for s in statuses.values()),
                "remaining": remaining, "unresolved": sorted(name for name, s in statuses.items() if s == "unresolved"),
                "ready_to_finish": not remaining and bool(self.views), "completion_policy": COMPLETION_POLICY}

    def fit_require(self, parameters):
        """Add {parameter_name: description} for any new task or object."""
        self._mutable()
        if not isinstance(parameters, dict) or not parameters or len(parameters) > 100:
            raise ValueError("Provide one to 100 named requirements")
        for name, description in parameters.items():
            self._name(name)
            if not isinstance(description, str) or not 1 <= len(description) <= 2000:
                raise ValueError("A concise description is required")
            if name in self.requested and description != self.requested[name]:
                raise ValueError("Existing requirements cannot be weakened or redefined; add a new parameter instead")
        self.requested.update(parameters)
        self._save()
        return self.fit_requirements()

    def capture(self, robo, cameras=("cam_head", "cam_left_wrist", "cam_right_wrist")):
        self._mutable()
        bundle = observation_bundle(robo, cameras=cameras)
        result = {}
        for camera, view in bundle["views"].items():
            ident = f"view_{len(self.views) + 1:04d}"
            # Reconstruct an allowlist, never persist arbitrary observation metadata.
            calibration = copy.deepcopy({key: view["calibration"][key] for key in ("shape", "intrinsic_matrix", "extrinsic_matrix")})
            camera_arrays(calibration)
            if list(calibration["shape"]) != list(view["rgb"].shape):
                raise ValueError("Calibration shape differs from original RGB")
            path = self.root / (ident + ".png")
            Image.fromarray(view["rgb"]).save(path)
            item = {"camera": camera, "step": bundle["step"], "calibration": calibration,
                    "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
                    "rgb_sha256": view["rgb_sha256"], "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "image": path.relative_to(self.workspace).as_posix()}
            self.views[ident] = item
            atomic_json(self.root / (ident + ".json"), item)
            result[ident] = item
        self._save()
        return copy.deepcopy(result)

    def fit_image(self, view_id):
        """Return saved original RGB; display(robo.fit_image(view_id)) to inspect."""
        view = self.views[view_id]
        path = self.workspace / view["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != view["png_sha256"]:
            raise ValueError("Saved RGB evidence has changed")
        if json.loads((self.root / (view_id + ".json")).read_text()) != view:
            raise ValueError("Saved calibration evidence has changed")
        return np.asarray(Image.open(path).convert("RGB"))

    def fit_triangulate(self, name, observations):
        """Fit a named static landmark from [{view_id, uv:[col,row], sigma_px}, ...].

        Uses only saved RGB pixel selections and camera calibration. Rejects weak
        parallax, inconsistent correspondences, points behind cameras and duplicate
        views. Returns xyz, reprojection residuals and conditional uncertainty.
        """
        self._mutable(); self._name(name)
        if name in self.views or name in self.landmarks:
            raise ValueError("Use a new landmark name for every fit")
        result = triangulate(self.views, copy.deepcopy(observations))
        self.landmarks[name] = copy.deepcopy(result)
        self._save()
        return result

    def _evidence_views(self, evidence):
        if not isinstance(evidence, list) or not evidence or len(evidence) > 256:
            raise ValueError("Cite saved view IDs and/or accepted landmark names")
        ids = set()
        for item in evidence:
            if item in self.views:
                ids.add(item)
            elif item in self.landmarks and self.landmarks[item]["accepted"]:
                ids.update(ob["view_id"] for ob in self.landmarks[item]["observations"])
            else:
                raise ValueError("Unknown evidence or rejected landmark: " + str(item))
        for ident in ids:
            self.fit_image(ident)  # validate retained original bytes
        return sorted(ids)

    def fit_estimate(self, name, value, unit, uncertainty, evidence, method, derivation):
        """Record an RGB-derived parameter with value, uncertainty and derivation.

        value/uncertainty accept scalars, arrays or per-instance dicts. Uncertainty
        must be scalar or match the value's complete array/mapping structure. Units are
        m, m2, rad, deg, px, count, dimensionless or mixed (describe every component).
        Metric/mixed estimates require distinct saved camera centers. evidence is
        a list of view IDs or accepted landmarks. method describes object/frame,
        assumptions and limitations; derivation contains the actual calculation
        or fitting code.
        """
        self._mutable(); self._name(name)
        if name not in self.requested:
            raise ValueError("Declare this parameter using fit_require first")
        if unit not in {"m", "m2", "rad", "deg", "px", "count", "dimensionless", "mixed"}:
            raise ValueError("Use a supported unit; specify mixed units in method")
        if any(not isinstance(s, str) or not 10 <= len(s.strip()) <= 20000 for s in (method, derivation)):
            raise ValueError("Document the measurement method AND actual derivation")
        value = numeric(value)
        uncertainty = numeric(uncertainty, nonnegative=True)
        if not isinstance(uncertainty, float) and not same_structure(value, uncertainty):
            raise ValueError("Uncertainty must be scalar or have the same structure as value")
        ids = self._evidence_views(evidence)
        if unit in {"m", "m2", "rad", "deg", "mixed"}:
            centers = np.array([camera_arrays(self.views[i]["calibration"])[1][:3, 3] for i in ids])
            if len(ids) < 2 or np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max() < 0.01:
                raise ValueError("Metric estimates need two distinct camera centers")
        record = {"status": "estimated", "value": value, "unit": unit,
                  "uncertainty": uncertainty, "evidence": evidence, "view_ids": ids,
                  "method": method, "derivation": derivation, "source": "rgb_estimate"}
        self.estimates[name] = copy.deepcopy(record)
        self._save()
        return record

    def fit_unresolved(self, name, reason):
        """Save a NONTERMINAL investigation note; this parameter still blocks finish.

        Use while investigating missing views, failed fits or ambiguous geometry.
        Continue collecting evidence or trying another method. Only fit_estimate
        or a justified fit_impossible claim can resolve the outstanding parameter.
        """
        self._mutable()
        if name not in self.requested or not isinstance(reason, str) or not 10 <= len(reason.strip()) <= 4000:
            raise ValueError("Name a requested parameter and explain why it is unresolved")
        self.estimates[name] = {"status": "unresolved", "reason": reason}
        self._save()
        return dict(self.estimates[name])

    def fit_impossible(self, name, reason, category, evidence, attempts, justification, alternatives):
        """Record a reviewable claim that permitted observations cannot identify a value.

        This is a last resort, never a shortcut for a failed fit, missing estimator
        definition, low confidence, or running out of time/actions/tokens. Exhaust
        feasible methods and views first. A noisy but identifiable quantity needs
        an estimate with honest uncertainty, not an impossibility claim.

        category: non_identifiable, observation_inaccessible, or outside_permitted_inputs.
        evidence: saved view IDs or accepted landmarks supporting the claim.
        attempts: at least two distinct {method, result} records describing actual
        investigations (including analyses); two failures alone never suffice.
        justification: explain why ALL permitted observations leave the quantity
        undetermined. For non-identifiability give distinct values/models with
        identical allowed observations. For inaccessible observations establish
        why all relevant safe views are unavailable, not merely untried.
        alternatives: nonempty [{approach, blocked_by}] explaining why remaining
        plausible approaches cannot resolve the quantity under the input/action
        constraints. Discuss the missing experiment or assumption explicitly.

        Each explanation must be concrete and grounded in the saved evidence.
        The host validates structure/evidence integrity, not the physical proof.
        This claim remains review_required and is not a numerical measurement.
        """
        self._mutable()
        if name not in self.requested:
            raise ValueError("Name a requested parameter")
        if not isinstance(category, str) or category not in IMPOSSIBILITY_CATEGORIES:
            raise ValueError("Use a supported impossibility category; resource exhaustion is incomplete")
        for label, value, minimum in (("reason", reason, 10), ("justification", justification, 40)):
            if not isinstance(value, str) or not minimum <= len(value.strip()) <= 20000:
                raise ValueError("Provide a concrete " + label + " for impossibility within permitted observations")
        for label, records, fields, minimum in (("attempts", attempts, {"method", "result"}, 2),
                                                ("alternatives", alternatives, {"approach", "blocked_by"}, 1)):
            if not isinstance(records, list) or not minimum <= len(records) <= 100:
                raise ValueError(f"Document at least {minimum} {label} with concrete outcomes or blockers")
            for record in records:
                if (not isinstance(record, dict) or set(record) != fields
                        or any(not isinstance(s, str) or not 10 <= len(s.strip()) <= 20000 for s in record.values())):
                    raise ValueError(f"Each {label} record needs detailed {sorted(fields)}")
        if len({r["method"].strip().casefold() for r in attempts}) != len(attempts):
            raise ValueError("Document distinct attempted methods, not repeated attempts with the same description")
        ids = self._evidence_views(evidence)
        record = {"status": "impossible", "reason": reason, "category": category,
                  "evidence": evidence, "view_ids": ids, "attempts": attempts,
                  "justification": justification, "alternatives": alternatives,
                  "source": "agent_rgb_impossibility_claim", "review_required": True}
        self.estimates[name] = copy.deepcopy(record)
        self._save()
        return copy.deepcopy(record)

    def fit_finish(self):
        """Freeze only when ALL parameters are estimated or supported as impossible.

        fit_progress lists outstanding work. Missing entries and fit_unresolved
        notes block completion. Lack of time, tokens or actions means incomplete,
        not impossible. Impossibility claims require review and are not
        numerical measurements. This ends calibration, not a task-success verdict;
        no values are installed into production policies automatically.
        """
        self._mutable()
        progress = self.fit_progress()
        if progress["remaining"]:
            raise ValueError("Continue investigating outstanding parameters: " + ", ".join(progress["remaining"])
                             + ". Supply an RGB estimate or evidence-backed fit_impossible claim; unresolved notes cannot finish.")
        if not self.views:
            raise ValueError("Inspect and retain RGB evidence before finishing")
        for ident in self.views:
            self.fit_image(ident)
        self.finished = True
        try:
            self._save()
        except BaseException:
            self.finished = False
            raise
        return {"status": "complete", "estimated": progress["estimated"],
                "impossible": progress["impossible"], "unresolved": 0}
