"""Episode-local spatial temporal memory. Task stages and reward truth are excluded."""
from __future__ import annotations

import copy
from collections import Counter
import json
import math
import os
import shutil
from pathlib import Path
import uuid

import numpy as np

from .stmem_geometry import align_box, angular_velocity, rotation


def quality(observation):
    # Completeness is an explicit assessment, not mask area or detection confidence.
    rank = {"occluded": 0,"partial": 0, "unknown": 0, "complete": 1}
    return (rank[observation.get("completeness", "unknown")], float(observation.get("confidence", 0)))


def similarity(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not a.size:
        return 0.
    return float(np.clip(a @ b / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))


def exemplar_quality(item):
    complete, confidence = quality(item)
    return complete, float(item.get("frontality") or 0), confidence


class SpatialMemory:
    def __init__(self, directory, *, max_gap_s=2., match_threshold=.65, ambiguity_margin=.12, require_confirmation=False, observation_confidence_threshold=.6,foundation_only=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entities = {}
        self.initial_review_status = "not_requested" if require_confirmation else "legacy"
        self.require_confirmation = require_confirmation
        self.foundation_only=foundation_only
        if foundation_only:self.initial_review_status="not_required_foundation_models"
        from .stmem_lifecycle import IdentityLifecycle
        self.completeness_reviews={}
        self.identity = IdentityLifecycle(self,observation_confidence_threshold)
        self.sample_index = -1
        self.instruction = ""
        self.working_set={"phase":"unknown","entity_ids":[],"reason":"Not yet selected by LLM","source":"LLM_pending"}
        self.max_gap_s, self.match_threshold, self.ambiguity_margin = max_gap_s, match_threshold, ambiguity_margin

    def log(self, kind, payload):
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "payload": payload}, allow_nan=False) + "\n")

    @staticmethod
    def image_continuity(obs, entity):
        """Same-camera 2D evidence remains useful when monocular metric depth jitters."""
        state = entity.get("tracking_current") or entity["state"]
        old = entity["best_observation"]
        if obs.get("camera") != old.get("camera") or obs.get("image_size") != state.get("image_size"):
            return None
        prediction=entity.get('tracking_prediction',{})
        if prediction.get('time')==obs['time'] and prediction.get('camera')==obs.get('camera'):
            state={**state,'bbox_2d':prediction['bbox_2d']}
        a, b = obs.get("bbox_2d"), state.get("bbox_2d")
        if a is None or b is None:return None
        a,b = np.asarray(a,float),np.asarray(b,float)
        if a.shape != (4,) or b.shape != (4,):return None
        sa,sb = a[2:]-a[:2],b[2:]-b[:2]
        if min(*sa,*sb)<=0:return None
        inter = float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
        aa,ab = float(sa.prod()),float(sb.prod())
        iou = inter/max(1.,aa+ab-inter)
        containment = inter/max(1.,min(aa,ab))
        distance = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
        proximity = math.exp(-distance/max(1.,min(np.linalg.norm(sa),np.linalg.norm(sb))*.5))
        if min(aa,ab)/max(aa,ab)<.2:return 0.
        # A partial mask contained in the previous footprint is positive identity
        # evidence; shrinking visible extent must not manufacture another object.
        return max(iou,containment*proximity**.25)

    def merged_mask_candidates(self, obs, now):
        """A SAM3 mask spanning two distinct known instances is not a new object."""
        a = obs.get("bbox_2d")
        if a is None:return []
        a = np.asarray(a,float);area = float(np.prod(a[2:]-a[:2]));covered = []
        for eid,e in self.entities.items():
            old,state = e["best_observation"],e["state"]
            if obs["frame_id"]!=state["frame_id"] or obs["label"].casefold() not in e["aliases"] or obs.get("camera")!=old.get("camera") or obs.get("image_size")!=state.get("image_size"):
                continue
            if now-state["time"]>self.max_gap_s or state["visibility"]=="out_of_view":continue
            b = state.get("bbox_2d")
            if b is None:continue
            b=np.asarray(b,float);barea=float(np.prod(b[2:]-b[:2]))
            inter=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
            if barea>0 and barea<.75*area and inter/barea>=.7:covered.append((eid,b,barea))
        for i,(eid,b,ba) in enumerate(covered):
            for other,c,ca in covered[i+1:]:
                inter=float(np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])).prod())
                if inter/max(1.,ba+ca-inter)<.2:return [eid,other]
        return []

    def _score(self, obs, entity):
        state, old = entity["state"], entity["best_observation"]
        if obs["frame_id"] != state["frame_id"] or obs["label"].casefold() not in entity["aliases"]:
            return 0.
        dt = max(0., obs["time"] - state["time"])
        predicted = np.array(state["box"]["center"]) + np.array(state["velocity_m_s"]) * min(dt, self.max_gap_s)
        distance = np.linalg.norm(np.array(obs["box"]["center"]) - predicted)
        gate = .10 + min(dt, self.max_gap_s) * .15 + state["position_uncertainty_m"]
        continuity = self.image_continuity(obs, entity)
        if distance > gate and state["visibility"] == "visible" and dt <= self.max_gap_s and (continuity is None or continuity < .7):
            return 0.
        # Compare best complete/frontal exemplars, not the immediately previous occluded crop.
        exemplars = sorted(entity["patterns"], key=exemplar_quality, reverse=True)[:5]
        appearance = max((similarity(obs.get("feature", []), e.get("feature", [])) for e in exemplars), default=0.)
        a, b = sorted(obs["box"]["dimensions"]), sorted(old["box"]["dimensions"])
        shape = float(np.exp(-np.linalg.norm(np.log((np.array(a)+.005)/(np.array(b)+.005)))))
        motion = math.exp(-distance / max(.01, gate))
        if entity.get('lifecycle')=='dormant' or dt>self.max_gap_s:
            return .85*appearance+.15*shape # Retrieval score only; return still requires identity evidence.
        if continuity is not None:
            return .70*continuity + .15*appearance + .10*motion + .05*shape
        return .55 * appearance + .30 * motion + .15 * shape

    def update(self, observations, time, *, reidentify=None, visibility=None, sample_index=None):
        """One timestamp/view batch, globally one-to-one matched. Strict mode stages unknown identities.

        reidentify receives actual crop references and ranked candidates. It returns
        an entity_id or None; callers must enforce its own deadline. Re-entry after a
        gap ALWAYS needs this judgment, including unexpected entry directions.
        """
        from scipy.optimize import linear_sum_assignment
        self.sample_index = self.sample_index+1 if sample_index is None else sample_index
        observations = copy.deepcopy(observations)
        for obs in observations:
            if abs(float(obs["time"]) - time) > 1e-6:
                raise ValueError("observation batch timestamps differ")
            rotation(obs["box"]["R"])
            if not np.isfinite(obs["box"]["center"]).all() or not np.isfinite(obs["box"]["dimensions"]).all() or min(obs["box"]["dimensions"]) < 0:
                raise ValueError("invalid metric box")
            if not 0 <= obs["confidence"] <= 1 or not math.isfinite(time):
                raise ValueError("invalid observation confidence/time")
            obs.setdefault("observation_id", uuid.uuid4().hex)
        from .stmem_completeness import apply as apply_completeness
        for obs in observations:apply_completeness(self,obs)
        ids = list(self.entities)
        scores = np.array([[self._score(o, self.entities[e]) for e in ids] for o in observations])
        group_masks = {i:self.merged_mask_candidates(o,time) for i,o in enumerate(observations)}
        for i,covered in group_masks.items():
            if covered:scores[i,:]=0.
        assignments = {}
        if observations and ids:
            # Dummy columns represent explicitly unmatched observations.
            cost = np.concatenate([1-scores, np.full((len(observations), len(observations)), 1-self.match_threshold)], axis=1)
            for i, j in zip(*linear_sum_assignment(cost)):
                if j < len(ids) and scores[i, j] >= self.match_threshold:
                    assignments[i] = ids[j]
        full_ids={eid for i,eid in assignments.items() if self.identity.full_for(observations[i],eid)}
        used, results = set(), []
        self.identity.expire(time)
        for i, obs in enumerate(observations):
            if group_masks[i]:
                self.log("unassigned_observation", {"reason":"mask_spans_known_instances",
                    "observation":obs,"candidate_entities":group_masks[i]})
                results.append(None)
                continue
            if self.require_confirmation and obs.get("completeness")!="complete":
                eid_candidate=assignments.get(i)
                if not self.identity.full_for(obs,eid_candidate) and (obs.get("completeness") in {"partial","occluded"} or eid_candidate is not None or self.identity.nearby(obs)):
                    forced_owner=None
                    if eid_candidate and self.entities[eid_candidate].get('lifecycle')=='dormant':
                        ranked=sorted([{'entity_id':e,'score':float(scores[i,j]),
                            'patterns':sorted(self.entities[e]['patterns'],key=exemplar_quality,reverse=True)[:3],
                            'last_state':self.entities[e]['state']} for j,e in enumerate(ids) if scores[i,j]>.35],key=lambda c:-c['score'])
                        chosen=reidentify({'observation':obs,'candidates':ranked[:4],'unmatched_allowed':True}) if reidentify else None
                        if chosen not in {c['entity_id'] for c in ranked} or chosen in full_ids:
                            results.append(None);continue
                        forced_owner=chosen
                    owner=self.identity.attach_tentative(obs,full_ids,time,forced_owner=forced_owner)
                    results.append(owner)
                    continue
            if self.require_confirmation and self.identity.full_for(obs,assignments.get(i)) and obs.get("completeness")=="unknown":
                obs["completeness"]="complete"
                obs["completeness_source"]="vlm_identical_visual_evidence"
            candidates = sorted([{"entity_id": eid, "score": float(scores[i, j])} for j, eid in enumerate(ids) if scores[i, j] > .35], key=lambda v: -v["score"])
            eid = assignments.get(i)
            ambiguous = len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < self.ambiguity_margin
            if eid:
                last = self.entities[eid]["state"]
                continuity = self.image_continuity(obs,self.entities[eid])
                appearance = max((similarity(obs.get("feature",[]),p.get("feature",[]))
                    for p in self.entities[eid]["patterns"]),default=0.)
                local_return = (last["visibility"] != "out_of_view" and continuity is not None and
                    (time-last["time"] <= self.max_gap_s and continuity >= .7 or
                     continuity >= .9 and appearance >= .85))
                returning = (last["visibility"] != "visible" or time-last["time"] > self.max_gap_s) and not local_return
                if ambiguous or returning:
                    evidence = {"observation": obs, "candidates": [
                        {**c, "patterns": sorted(self.entities[c["entity_id"]]["patterns"], key=exemplar_quality, reverse=True)[:3],
                         "last_state": self.entities[c["entity_id"]]["state"]} for c in candidates[:4]],
                        "unmatched_allowed": True}
                    chosen = reidentify(evidence) if reidentify else None
                    eid = chosen if chosen in {c["entity_id"] for c in candidates} and chosen not in used else None
            if eid in used:
                eid = None
            pending = None
            if eid is None and self.require_confirmation:
                reason = "bbox_or_label_conflict" if self.identity.candidates(obs) else "unconfirmed_or_partial_observation"
                pending = self.identity.stage(obs,reason,time)
                if not (self.foundation_only and reason=="unconfirmed_or_partial_observation") and not self.identity.admit(pending,obs):
                    results.append(None)
                    continue
            match_continuity = self.image_continuity(obs,self.entities[eid]) if eid is not None else None
            created_new = eid is None
            if eid is None:
                eid = "entity_" + uuid.uuid4().hex[:12]
                self.entities[eid] = {"entity_id": eid, "aliases": [obs["label"].casefold()], "patterns": [],
                                      "history": [], "visibility_history": [], "revisions": [], "best_observation": copy.deepcopy(obs), "state": None,
                                      "description_status":"ready" if obs.get("description") else "unannotated" if self.foundation_only else "not_requested",
                                      "identity_status":"foundation_track" if self.foundation_only else "legacy_unverified"}
            entity = self.entities[eid]
            old = entity["state"]
            box = copy.deepcopy(obs["box"])
            dt = time-old["time"] if old else 0
            if dt < 0:
                raise ValueError("out-of-order observation; replay in timestamp order")
            alignment = np.eye(3)
            if old:
                r, dims, alignment = align_box(box["R"], box["dimensions"], old["box"]["R"])
                box.update(R=r.tolist(), dimensions=dims.tolist())
            velocity = ((np.array(box["center"])-np.array(old["box"]["center"]))/dt).tolist() if dt > 0 else [0., 0., 0.]
            omega = angular_velocity(old["box"]["R"], box["R"], dt).tolist() if dt > 0 else [0., 0., 0.]
            # Update measurements even when partial. Only shape/semantic templates obey quality dominance.
            if quality(obs) > quality(entity["best_observation"]):
                entity["best_observation"] = copy.deepcopy({**obs, "box": box})
                # A better rigid-object extent revises the model throughout the
                # trajectory; raw visible-surface measurements remain untouched.
                for historical in entity["history"]:
                    historical["model_dimensions"] = copy.deepcopy(box["dimensions"])
                entity["revisions"].append({"revision": len(entity["revisions"])+1,
                    "kind": "shape", "dimensions": copy.deepcopy(box["dimensions"]), "observation_id": obs["observation_id"]})
                for historical in entity["history"]:
                    historical["revision"] = len(entity["revisions"])
                self.log("shape_update", {"entity_id": eid, "observation_id": obs["observation_id"], "quality": quality(obs)})
            state = {"time": time, "frame_id": obs["frame_id"], "box": box,
                     "bbox_2d": copy.deepcopy(obs.get("bbox_2d")),
                     "image_size": copy.deepcopy(obs.get("image_size")),
                     "model_dimensions": entity["best_observation"]["box"]["dimensions"],
                     "velocity_m_s": velocity, "angular_velocity_rad_s": omega,
                     "visibility": "visible", "motion": "moving" if np.linalg.norm(velocity) > .01 or np.linalg.norm(omega) > .1 else "stationary",
                     "position_uncertainty_m": float(obs.get("position_uncertainty_m", .05)),
                     "orientation_ambiguous": obs.get("orientation_ambiguous", True),
                     "observed_or_predicted": "observed", "evidence": obs["observation_id"],
                     "geometry_provenance":copy.deepcopy(obs.get("geometry_provenance")),
                     "revision": len(entity["revisions"]), "box_frame_alignment":alignment.tolist()}
            # Yaw is a convenience only when the same upward face is stable; R
            # remains authoritative. Never claim this is a T's semantic heading.
            same_face = old is not None and np.dot(np.array(old["box"]["R"])[:, 2], np.array(box["R"])[:, 2]) > .99
            state["yaw_rad"] = float(np.arctan2(box["R"][1][0], box["R"][0][0])) if same_face and abs(box["R"][2][2]) > .99 and state["motion"] == "stationary" else None
            if not entity["best_observation"].get("description") and obs.get("description"):
                entity["best_observation"]["description"] = obs["description"]
                entity["best_observation"]["description_observation_id"] = obs["observation_id"]
            entity["state"] = state
            entity.pop("tracking_current",None)
            entity["tracking_support"]={"time":time,"kind":"complete_observation" if obs.get("completeness")=="complete" else "observation",
                "observation_ids":[obs["observation_id"]],"camera":obs.get("camera")}
            if obs.get("completeness")=="complete":
                entity["last_complete"]={"time":time,"observation_id":obs["observation_id"],"camera":obs.get("camera"),
                    "frame_id":obs["frame_id"],"box":copy.deepcopy(box),"bbox_2d":copy.deepcopy(obs.get("bbox_2d")),
                    "source":copy.deepcopy(obs.get("geometry_provenance"))}
            entity["lifecycle"] = "active"
            entity["visibility_history"].append({"time": time, "visibility": "visible"})
            entity["history"].append(copy.deepcopy(state))
            if obs.get("description") and not entity.get("description_request") and not entity.get("semantic_annotation") and (not entity.get("description_record") or
                    quality(obs) >= tuple(entity["description_record"]["quality"])):
                entity["description_status"] = "ready"
                entity["description_record"] = {"text":obs["description"], "quality":list(quality(obs)),
                    "observation_id":obs["observation_id"], "source":obs.get("description_source"), "time":time}
            entity["patterns"].append({k: copy.deepcopy(obs.get(k, 0. if k == "frontality" else "unknown" if k == "completeness" else None)) for k in ("time", "crop_path", "feature", "completeness", "confidence", "frontality", "observation_id")})
            used.add(eid)
            self.log("observation", {"entity_id": eid, "observation": obs, "association_candidates": candidates, "unmatched_allowed": True,
                "association": {"created_new":created_new,"ambiguous":ambiguous,
                    "image_continuity":match_continuity}})
            self.log("state", {"entity_id": eid, "state": state})
            if self.require_confirmation and self.identity.full_for(obs,eid):self.identity.merge_tentatives(eid,time)
            self.archive(entity, obs, time, candidates)
            if pending is not None:
                if self.foundation_only:
                    pending['status']='consumed';pending['entity_id']=eid
                    from .stmem_anchors import make_anchor
                    assets={key:f"frames/{self.sample_index:06d}/{name}" for key,name in (
                        ("bbox_crop_path","bbox-crop.png"),("crop_path","crop.png"),("mask_path","mask.png"),
                        ("geometry_path","geometry.npz"),("calibration_path","calibration.json"))
                        if obs.get(key) and Path(obs[key]).is_file()}
                    entity.setdefault('initial_anchor',make_anchor(obs,known_at=time,assets=assets))
                    self.archive_summary(entity)
                else:self.identity.consume(pending,entity,time)
            results.append(eid)
        for eid, entity in self.entities.items():
            if eid not in used and entity["state"]:
                status = "occluded" if self.identity.has_tentative(eid) else (visibility or {}).get(eid, "unknown")
                if status not in {"unknown", "occluded", "out_of_view"}:
                    raise ValueError("invalid unobserved visibility")
                entity["lifecycle"] = "dormant" if status=="out_of_view" or time-(entity.get("tracking_current") or entity["state"])["time"]>self.max_gap_s else "active"
                entity["state"]["visibility"] = status
                entity["visibility_history"].append({"time": time, "visibility": status})
                self.log("visibility", {"entity_id": eid, "time": time, "visibility": status})
                self.archive(entity, None, time, [])
        return results

    def apply_description(self, entity_id, item, source, now, *, error=None):
        """Commit one first-crop GPT result without changing geometry or old observations."""
        entity = self.entities[entity_id]
        if entity.get("description_status") == "ready":
            return False
        text = item.get("description") if isinstance(item,dict) else None
        if error or not isinstance(text,str) or not text.strip():
            entity["description_status"] = "failed"
            entity["description_error"] = error or "empty_description"
            self.log("description_failed", {"entity_id":entity_id,"reason":entity["description_error"],"source":source})
            self.archive_summary(entity)
            return False
        label, valid = item.get("label"), item.get("label_valid")
        if valid is not None and type(valid) is not bool:
            entity["description_status"] = "failed"
            entity["description_error"] = "invalid_label_validation"
            self.archive_summary(entity)
            return False
        if isinstance(label,str) and label.strip() and valid is not None:
            label = label.strip()[:80]
            entity["semantic_label"] = label
            if label.casefold() not in entity["aliases"]:entity["aliases"].append(label.casefold())
        entity["label_validation"] = {"valid":valid,"label":entity.get("semantic_label"),"source":source}
        prior=entity["best_observation"].get("description","")
        proposed=text.strip();applied=[]
        if isinstance(prior,str) and prior.strip():
            text=prior.strip()
            # Existing appearance is immutable unless a scoped, instruction-backed
            # replacement is supplied. A full rewritten description cannot erase it.
            for patch in item.get("description_corrections",[]) if isinstance(item.get("description_corrections",[]),list) else []:
                if not isinstance(patch,dict) or patch.get("attribute") not in {"number","color","shape","marking"}:continue
                old,new,phrase=(patch.get(k) for k in ("old_text","new_text","instruction_phrase"))
                if not all(isinstance(v,str) and v.strip() for v in (old,new,phrase)):continue
                if patch.get("supported_by_rgb") is not True or phrase not in self.instruction or new not in phrase or old not in text:continue
                text=text.replace(old,new,1);applied.append(copy.deepcopy(patch))
        entity["description_record"] = {"text":text.strip()[:180], "source":source,
            "observed_text":prior,"proposed_text":proposed,"applied_attribute_corrections":applied,
            "observation_id":source.get("observation_id"), "time":source.get("time"),
            "available_at":now, "quality":list(quality(entity["best_observation"]))}
        for key in ("instruction_mentioned","interactable","avoid_interaction"):
            value=item.get(key)
            entity[key]=value if type(value) is bool else None
        entity["semantic_annotation_source"] = copy.deepcopy(source)
        entity["description_status"] = "ready"
        self.log("description_ready", {"entity_id":entity_id,"description_record":entity["description_record"]})
        self.archive_summary(entity)
        phrase = item.get("instruction_label")
        if isinstance(phrase,str) and phrase:self.resolve(phrase,now=now)
        return True

    def archive(self, entity, observation, time, candidates, reason=None):
        """Portable per-entity sampled frames; raw observations stay immutable."""
        directory = self.directory / "entities" / entity["entity_id"]
        frame = directory / "frames" / f"{self.sample_index:06d}"
        frame.mkdir(parents=True, exist_ok=True)
        assets = {}
        if observation:
            for key, name in (("crop_path", "crop.png"), ("mask_path", "mask.png"),
                              ("bbox_crop_path", "bbox-crop.png"),
                              ("geometry_path", "geometry.npz"), ("calibration_path", "calibration.json")):
                source = Path(observation.get(key) or "__missing__")
                if source.is_file():
                    try:
                        os.link(source, frame/name)
                    except OSError:
                        shutil.copy2(source, frame/name)
                    assets[key] = str((frame/name).relative_to(directory))
        state = copy.deepcopy(entity.get("tracking_current") or entity["state"])
        if reason == "scheduled_skip" and not self.identity.has_tentative(entity["entity_id"]):
            state["visibility"] = "unknown"
        description = entity.get("description_record", {})
        row = {"schema":"oeh.stmem.object_frame.v1", "entity_id":entity["entity_id"],
               "sample_index":self.sample_index, "time":time, "last_observed_time":state["time"],
               "observed":observation is not None, "frame_id":state["frame_id"], "units":"m",
               "bbox":state["box"], "R":state["box"]["R"], "t":state["box"]["center"],
               "dim":state["box"]["dimensions"], "label":entity.get("context_label",entity.get("semantic_label",entity["aliases"][0])),
               "detector_label":entity["aliases"][0], "desc":description.get("text",entity["best_observation"].get("description", "")),
               "description_source":description.get("source"),
               "description_status":entity.get("description_status","not_requested"),
               **{k:entity.get(k) for k in ("identity_status","lifecycle","instruction_mentioned","interactable","avoid_interaction")},
               "label_validation":entity.get("label_validation"), "reason":reason, "state":state,
               "bbox_2d":observation.get("bbox_2d") if observation else None,
               "association_candidates":candidates, "unmatched_allowed":True, "assets":assets}
        # Absolute pod paths are provenance, not links required for offline viewing.
        row["observation"] = ({k:v for k,v in observation.items() if not k.endswith("_path")}
                              if observation else None)
        tmp = frame/"metadata.json.tmp"
        tmp.write_text(json.dumps(row,allow_nan=False,indent=2)+"\n", encoding="utf-8")
        tmp.replace(frame/"metadata.json")
        with (directory/"trajectory.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        self.archive_summary(entity)

    def archive_summary(self, entity):
        directory = self.directory / "entities" / entity["entity_id"]
        directory.mkdir(parents=True, exist_ok=True)
        public=self.query("current",entity["entity_id"],now=(entity.get("tracking_current") or entity["state"])["time"])
        summary = {"schema":"oeh.stmem.entity.v1", "entity_id":entity["entity_id"],
                   "label":public["label"], "detector_label":entity["aliases"][0], "aliases":entity["aliases"],
                    "camera":entity["best_observation"].get("camera"),
                   "desc":public["description"],
                   "description_record":entity.get("description_record"),
                   "semantic_annotation":entity.get("semantic_annotation"),
                   "semantic_annotation_history":entity.get("semantic_annotation_history",[]),
                   "identity_status":entity.get("identity_status","legacy_unverified"),
                   "lifecycle":entity.get("lifecycle","active"),
                   **{k:entity.get(k) for k in ("instruction_mentioned","interactable","avoid_interaction","semantic_annotation_source","confirmation","first_complete_observation","initial_anchor","last_complete","tracking_support","tracking_history","field_status","field_events","tracking_observations")},
                   "description_status":entity.get("description_status","not_requested"),
                   "description_request":entity.get("description_request"), "description_error":entity.get("description_error"),
                   "label_validation":entity.get("label_validation"), "label_history":entity.get("label_history",[]),
                   "bbox":public["box"], "R":public["box"]["R"],
                   "t":public["box"]["center"], "dim":public["box"]["dimensions"],
                   "current":public,"raw_last_measurement":entity["state"], "corrected_history":entity["history"],
                   "visibility_history":entity["visibility_history"], "revisions":entity["revisions"],
                   "trajectory":"trajectory.jsonl", "frame_cadence":"SAM3 perception samples, not every physics tick"}
        tmp = directory/"entity.json.tmp"
        tmp.write_text(json.dumps(summary,allow_nan=False,indent=2)+"\n", encoding="utf-8")
        tmp.replace(directory/"entity.json")

    def revise_frame(self, entity_id, old_from_new, *, reason, origin_shift_old=None):
        """Change canonical box axes for ALL historical poses, never erase physical motion.

        A full SO(3) canonical-frame correction is supported. With arbitrary Q,
        dimensions conservatively enclose the old box's corners in the new frame;
        only signed permutations preserve a tight enclosure exactly. A measured
        canonical-origin correction can also be propagated through the trajectory.
        New physical motion arrives via update(), never this method. Raw observations
        remain immutable in history.jsonl; corrected queries carry revision.
        """
        q = rotation(old_from_new)
        shift = np.zeros(3) if origin_shift_old is None else np.asarray(origin_shift_old, float)
        if shift.shape != (3,) or not np.isfinite(shift).all() or not reason:
            raise ValueError("revision requires a finite origin shift and reason")
        entity = self.entities[entity_id]
        revision = {"revision": len(entity["revisions"])+1, "old_from_new": q.tolist(), "origin_shift_old": shift.tolist(), "reason": reason}
        entity["revisions"].append(revision)
        for state in entity["history"] + [entity["state"]]:
            box = state["box"]
            box["center"] = (np.array(box["center"])+np.array(box["R"]) @ shift).tolist()
            box["R"] = (np.array(box["R"]) @ q).tolist()
            box["dimensions"] = (abs(q).T @ box["dimensions"]+2*abs(q.T @ shift)).tolist()
            state["model_dimensions"] = (abs(q).T @ state["model_dimensions"]+2*abs(q.T @ shift)).tolist()
            box["revision_extent"] = "conservative_reenclosure"
            state["yaw_rad"] = None
            state["revision"] = revision["revision"]
        best = entity["best_observation"]["box"]
        best["center"] = (np.array(best["center"])+np.array(best["R"]) @ shift).tolist()
        best["R"] = (np.array(best["R"]) @ q).tolist()
        best["dimensions"] = (abs(q).T @ best["dimensions"]+2*abs(q.T @ shift)).tolist()
        # Changing an object's origin can change its translational velocity during
        # rotation. Recompute corrected history instead of keeping inconsistent rates.
        for previous, current in zip(entity["history"], entity["history"][1:]):
            dt = current["time"]-previous["time"]
            if dt > 0:
                current["velocity_m_s"] = ((np.array(current["box"]["center"])-previous["box"]["center"])/dt).tolist()
                current["angular_velocity_rad_s"] = angular_velocity(previous["box"]["R"], current["box"]["R"], dt).tolist()
        if entity["history"]:
            entity["state"].update(velocity_m_s=entity["history"][-1]["velocity_m_s"], angular_velocity_rad_s=entity["history"][-1]["angular_velocity_rad_s"])
        self.log("frame_revision", {"entity_id": entity_id, **revision})
        self.archive_summary(entity)

    def set_instruction(self, instruction, now=None):
        instruction = instruction if isinstance(instruction,str) else ""
        if instruction != self.instruction:
            self.instruction = instruction
            for entity in self.entities.values():
                previous = entity.pop("context_label",None)
                if previous:
                    entity.setdefault("label_history",[]).append({"label":entity["aliases"][0],
                        "previous_label":previous,"reason":"instruction_changed",
                        "time":entity["state"]["time"] if now is None else now})
                    self.archive_summary(entity)

    def resolve(self, text, *, now=None):
        text = text.strip().casefold() if isinstance(text,str) else ""
        if not text:
            return {"best":None,"status":"not_observed","alternatives":[],"unmatched":{"allowed":True}}
        candidates = []
        for eid, entity in self.entities.items():
            names = entity["aliases"] + [eid.casefold()] + ([entity["context_label"].casefold()] if entity.get("context_label") else [])
            description = entity.get("description_record",{}).get("text",entity["best_observation"].get("description", "")).casefold()
            score = max([1. if text == n else .75 if text in n or n in text else 0. for n in names] + [.9 if text in description else 0.])
            if score:
                candidates.append({"entity_id": eid, "score": score})
        candidates.sort(key=lambda x: (-x["score"], x["entity_id"]))
        decisive = bool(candidates) and (len(candidates) == 1 or candidates[0]["score"]-candidates[1]["score"] >= self.ambiguity_margin)
        changed = False
        start = self.instruction.casefold().find(text)
        if start >= 0 and text.isascii():
            before = self.instruction[start-1:start] if start else ""
            after = self.instruction[start+len(text):start+len(text)+1]
            if before and (before.isalnum() or before == "_") or after and (after.isalnum() or after == "_"):
                start = -1
        if decisive and start >= 0:
            phrase = self.instruction[start:start+len(text)]
            entity = self.entities[candidates[0]["entity_id"]]
            changed = entity.get("context_label") != phrase
            entity["context_label"] = phrase
            if changed:
                event = {"label":phrase,"instruction":self.instruction,"time":entity["state"]["time"] if now is None else now,
                         "source":"decisive_instruction_phrase_match","candidate":candidates[0]}
                entity.setdefault("label_history",[]).append(event)
                self.log("instruction_label",{"entity_id":entity["entity_id"],**event})
                self.archive_summary(entity)
        return {"best": candidates[0] if candidates else None, "status": "resolved" if decisive else "ambiguous" if candidates else "not_observed",
                "context_label_changed":changed,
                "alternatives": candidates[1:], "unmatched": {"allowed": True}, "scores_are": "ranking_scores_not_probabilities"}

    def query(self, operation, entity_id=None, start=None, end=None, text=None, now=None):
        if operation == "resolve":
            return self.resolve(text or "",now=now)
        entity = self.entities.get(entity_id)
        if entity is None:
            return {"status": "not_observed", "entity_id": entity_id}
        if operation == 'initial':
            return {'entity_id':entity_id,'initial_anchor':copy.deepcopy(entity.get('initial_anchor')),
                'status':'ok' if entity.get('initial_anchor') else 'not_observed'}
        if operation == "current":
            state = copy.deepcopy(entity.get("tracking_current") or entity["state"])
            state["age_s"] = max(0, (now if now is not None else state["time"])-state["time"])
            if state["age_s"] > 0 and state["box"].get("center") is not None and state.get("velocity_m_s") is not None:
                state["predicted_center"] = (np.array(state["box"]["center"])+np.array(state["velocity_m_s"])*min(state["age_s"], self.max_gap_s)).tolist()
                state["prediction_uncertainty_m"] = state["position_uncertainty_m"] + .15*state["age_s"]
                state["motion"] = "unknown"
            visibility = entity["visibility_history"][-1] if entity["visibility_history"] else {"time":state["time"],"visibility":state["visibility"]}
            at = now if now is not None else visibility["time"]
            state["visibility_age_s"] = max(0,at-visibility["time"])
            state["visibility_at_last_update"] = visibility["visibility"]
            if state["visibility_age_s"] > 0:
                state["visibility"] = "unknown"
            if self.identity.has_tentative(entity_id):
                state["visibility"]="occluded"
                state["occlusion_reason"]="pending_tentative_merge"
            if self.require_confirmation:
                state['bbox_status']='observed' if state['age_s']==0 else 'last_observed'
                state['bbox_observed_at']=state['time']
                if state['age_s']>0 or state['visibility']!='visible' or self.foundation_only and not entity.get('last_complete'):
                    state['box']['center']=None;state['box']['R']=None
                    state.pop('predicted_center',None);state.pop('prediction_uncertainty_m',None)
                    state['motion']='unknown'
                    if self.foundation_only and not entity.get('last_complete'):state['model_dimensions']=None
                    state['velocity_m_s']=None;state['angular_velocity_rad_s']=None;state['yaw_rad']=None
                    state['observed_or_predicted']='partial_observation' if state.get('evidence') in entity.get('tracking_support',{}).get('observation_ids',[]) and entity.get('tracking_current') else 'unknown'
                    state['pose_status']={'center':'unknown','R':'unknown','dimensions':'last_complete' if entity.get('last_complete') else 'unknown'}
                    state['box']['dimensions']=copy.deepcopy(entity.get('last_complete',{}).get('box',{}).get('dimensions'))
                complete=entity.get('last_complete')
                state['dimensions_observed_at']=complete['time'] if complete else None
                if state['age_s']>0 and entity.get('tracking_current') and at>visibility['time']:state['visibility']='unknown'
            if self.foundation_only and state.get("orientation_ambiguous"):
                state["box"]["R"]=None
                state["angular_velocity_rad_s"]=None;state["yaw_rad"]=None
                state.setdefault("pose_status",{})["R"]="unknown"
            result = {"entity_id": entity_id, "label":entity.get("context_label",entity.get("semantic_label",entity["aliases"][0])),
                    "detector_label":entity["aliases"][0], "aliases":entity["aliases"],
                    "camera":entity["best_observation"].get("camera"),
                    "description":entity.get("description_record",{}).get("text",entity["best_observation"].get("description", "")),
                    "description_source":entity.get("description_record",{}).get("source"),
                    "semantic_annotation":copy.deepcopy(entity.get("semantic_annotation")),
                    "description_status":entity.get("description_status","not_requested"),
                    "label_validation":entity.get("label_validation"),
                    "field_status":copy.deepcopy(entity.get("field_status",{})),
                    "tracking_revision":len(entity.get("tracking_observations",[])),
                    "last_complete":copy.deepcopy(entity.get("last_complete")),
                    "tracking_support":copy.deepcopy(entity.get("tracking_support")),
                    "identity_status":entity.get("identity_status","legacy_unverified"),
                    "lifecycle":entity.get("lifecycle","active"),
                    **{k:entity.get(k) for k in ("instruction_mentioned","interactable","avoid_interaction")}, **state}
            for field,record in entity.get("field_status",{}).items():
                if record["status"]!="deleted":continue
                target=result;parts=field.split(".")
                for part in parts[:-1]:
                    if not isinstance(target.get(part),dict):target[part]={}
                    target=target[part]
                target[parts[-1]]="deleted"
                for derived in {"box.center":["predicted_center"],"box.R":["yaw_rad"],"box.dimensions":["model_dimensions"]}.get(field,[]):
                    if derived in result:result[derived]="deleted"
            result['current']={k:copy.deepcopy(result[k]) for k in state if k in result}
            return result
        history = entity["history"]
        if operation == "at":
            if start is None:
                raise ValueError("at requires start timestamp")
            previous = [s for s in history if s["time"] <= start]
            visibility = [s for s in entity["visibility_history"] if s["time"] <= start]
            visible_state=copy.deepcopy(previous[-1]) if previous else None
            status_at={}
            for event in entity.get("field_events",[]):
                if event["time"]<=start:status_at[event["field"]]=event
            if visible_state:
                for field,event in status_at.items():
                    if event["status"]!="deleted":continue
                    target=visible_state;parts=field.split(".")
                    for part in parts[:-1]:target=target.setdefault(part,{})
                    target[parts[-1]]="deleted"
            tracking=[s for s in entity.get('tracking_history',[]) if s['time']<=start]
            tracking_state=copy.deepcopy(tracking[-1]) if tracking and (not previous or tracking[-1]['time']>previous[-1]['time']) else None
            return {"entity_id": entity_id, "requested_time": start,"last_tracking_state":tracking_state, "last_observed_state": visible_state,"field_status_at":copy.deepcopy(status_at),
                    "visibility_at": copy.deepcopy(visibility[-1]) if visibility else None,
                    "exact_observation": bool(previous and previous[-1]["time"] == start),"exact_tracking_observation":bool(tracking_state and tracking_state["time"]==start), "interpolated": False}
        if operation == "trajectory":
            selected = [s for s in history if (start is None or s["time"] >= start) and (end is None or s["time"] <= end)]
            next_time = selected[128]["time"] if len(selected) > 128 else None
            return {"entity_id": entity_id, "states": copy.deepcopy(selected[:128]),
                    "next_start": next_time, "truncated": next_time is not None,
                    "visibility_events": copy.deepcopy([s for s in entity["visibility_history"] if (start is None or s["time"] >= start) and (end is None or s["time"] <= end) and (next_time is None or s["time"] < next_time)])}
        raise ValueError("unknown spatial memory operation")

    def mark_field_deleted(self,entity_id,field,*,reason,now):
        if field not in {"description","label","bbox_2d","box.R","box.center","box.dimensions","velocity_m_s","angular_velocity_rad_s"}:
            raise ValueError("unsupported deletable field")
        entity=self.entities[entity_id]
        entity.setdefault("field_status",{})[field]={"status":"deleted","reason":reason,"time":now}
        entity.setdefault("field_events",[]).append({"field":field,**entity["field_status"][field]})
        self.log("field_deleted",{"entity_id":entity_id,"field":field,"reason":reason,"time":now})
        self.archive_summary(entity)

    def restore_field(self,entity_id,field,*,reason,now):
        entity=self.entities[entity_id]
        entity.setdefault("field_status",{})[field]={"status":"active","reason":reason,"time":now}
        entity.setdefault("field_events",[]).append({"field":field,**entity["field_status"][field]})
        self.log("field_restored",{"entity_id":entity_id,"field":field,"reason":reason,"time":now})
        self.archive_summary(entity)

    def set_working_set(self,value,now,observation_id=None):
        self.working_set={**copy.deepcopy(value),"time":now,"observation_id":observation_id,"source":"LLM_report_not_verified_task_stage"}
        self.log("working_set",self.working_set)
        (self.directory/"working-set.json").write_text(json.dumps(self.working_set,indent=2)+"\n", encoding="utf-8")
        return {"status":"stored","working_set":copy.deepcopy(self.working_set)}

    def context(self, now, entity_ids=None):
        return {"schema": "oeh.stmem.context.v1", "snapshot_time": now,
                "working_set":copy.deepcopy(self.working_set),
                "initial_review_status":self.initial_review_status,
                "pending_observations":sum(p["status"] not in {"consumed","expired"} for p in self.identity.pending.values()),
                "tentative_tracking_count":sum(r["status"]=="active" for rows in self.identity.tentative.values() for r in rows),
                "entities": [self.query("current", eid, now=now) for eid in (entity_ids if entity_ids is not None else self.entities)],
                "description_states":dict(Counter(e.get("description_status","not_requested") for e in self.entities.values())),
                "rules": "Pending/failed descriptions are not validated semantic facts. Use entity_id. Unknown/unseen is not absent. Box axes are not semantic heading. No task-stage or reward truth is included."}
