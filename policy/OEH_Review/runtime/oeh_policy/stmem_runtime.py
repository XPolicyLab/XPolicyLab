"""Bounded worker transport and review-only spatial context/tool projection."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
from pathlib import Path
import signal
import uuid

import numpy as np


def validate_stmem(cfg):
    result = copy.deepcopy(cfg.get("stmem"))
    if not isinstance(result, dict):
        raise ValueError("review-stmem requires a stmem configuration")
    if cfg.get("benchmark", "robodojo") != "robodojo":
        raise ValueError("review-stmem is currently integrated with RoboDojo only")
    for key in ("python", "da3_checkpoint", "sam3_checkpoint", "calibration_file"):
        if not isinstance(result.get(key), str) or not Path(result[key]).is_absolute():
            raise ValueError(f"stmem.{key} requires an absolute deployed path")
    if result.get("inference_socket") and not Path(result["inference_socket"]).is_absolute():
        raise ValueError("stmem.inference_socket must be absolute")
    result.setdefault("perception_seconds", 20.)
    result.setdefault("startup_seconds", 60.)
    result.setdefault("camera", "cam_head")
    result.setdefault("labels", [])
    result.setdefault("device", "cuda")
    result.setdefault("perception_policy", "foundation-models")
    if result["perception_policy"] != "foundation-models":
        raise ValueError("unsupported perception policy; alternatives are design only")
    if any(k in result for k in ("visual_review_every", "visual_review_max_calls")):
        raise ValueError("visual review settings are retired; Memory cannot request LLM re-evaluation")
    # Historical experiment configs remain readable, but no helper options reach the worker.
    for key in ("semantic", "semantic_reasoning_effort", "discover_every", "describe_every",
                "description_mode", "description_timeout_s", "description_batch_size",
                "description_queue_limit", "identity_mode", "geometry_review_threshold"):
        result.pop(key, None)
    result.setdefault("observation_confidence_threshold",.6)
    if type(result["observation_confidence_threshold"]) not in (int,float) or not 0 < result["observation_confidence_threshold"] <= 1:raise ValueError("invalid observation confidence threshold")
    cfg["review"]["teacher"]["context_reset_every"]=0
    result.setdefault("max_objects", 24)
    result.setdefault("max_query_calls", 4)
    result.setdefault("perception_every", 1)
    if "task_protocol" in result and (result["task_protocol"] != "press_by_number-public-v1"
                                      or cfg["review"].get("task") != "press_by_number"):
        raise ValueError("unknown or mismatched STMem public task protocol")
    if "context_max_entities" in result:
        result.setdefault("context_max_chars", 24000)
        for key in ("context_max_entities", "context_max_chars"):
            if type(result[key]) is not int or result[key] < 1:
                raise ValueError(f"positive integer stmem.{key} required")
        if result["context_max_chars"] < 2000:
            raise ValueError("stmem context_max_chars must allow a compact spatial header")
    for key in ("perception_seconds", "startup_seconds"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or not 0 < result[key] <= (600 if key == "startup_seconds" else 60):
            raise ValueError(f"invalid stmem.{key}")
    for key in ("max_objects", "max_query_calls", "perception_every"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"positive integer stmem.{key} required")
    if result["camera"] not in {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
        raise ValueError("unknown stmem camera")
    if not isinstance(result["labels"], list) or not result["labels"] or not all(isinstance(s, str) and s.strip() for s in result["labels"]):
        raise ValueError("stmem.labels must list nonempty concepts")
    review = cfg["review"]
    if review["teacher"].get("channel", "python") != "python" or review.get("schedule") == "on_stall":
        raise ValueError("review-stmem currently requires the Python every-chunk teacher")
    if review["call_seconds"] + review["first_call_extra_seconds"] + 15 >= 120:
        raise ValueError("review-stmem policy call including transport reserve must be strictly below 120 seconds")
    if result["perception_seconds"] >= review["call_seconds"]-review["hold_reserve_s"]:
        raise ValueError("perception must leave a teacher execution window")
    if review["teacher"]["api"] not in {"codex_app_server", "openai_responses"}:
        raise ValueError("review-stmem requires codex_app_server or openai_responses")
    return result


def tool(original, *, task_tree=False):
    result = copy.deepcopy(original)
    from .stmem_queries import query_schemas
    result["parameters"] = {"type":"object", "properties":{"action":{"anyOf":[original["parameters"], *query_schemas()]}},
        "required":["action","stmem_working_set"],"additionalProperties":False}
    result["parameters"]["properties"]["stmem_working_set"]={"type":"object","additionalProperties":False,
        "properties":{"phase":{"type":"string"},"entity_ids":{"type":"array","items":{"type":"string"},"maxItems":32},"reason":{"type":"string"}},
        "required":["phase","entity_ids","reason"]}
    from .stmem_updates import update_schema
    props=result["parameters"]["properties"]
    props["action"]["anyOf"].append({"type":"null"})
    props["stmem_working_set"]={"anyOf":[props["stmem_working_set"],{"type":"null"}]}
    props["memory_update"]={"anyOf":[update_schema(),{"type":"null"}]}
    props["memory_query"]={"type":"array","maxItems":4,"items":{"type":"object","additionalProperties":False,
        "properties":{"query_id":{"type":"string"},"query":{"anyOf":query_schemas()}},"required":["query_id","query"]}}
    result["parameters"]["required"] += ["memory_update","memory_query"]
    if task_tree:
        from .task_tree_runtime import update_schema as task_update_schema, query_schema, binding_schema
        props["semmem_update"] = {"anyOf": [task_update_schema(), {"type": "null"}]}
        props["task_binding"] = {"anyOf": [binding_schema(), {"type": "null"}]}
        props["memory_query"]["items"]["properties"]["query"]["anyOf"].append(query_schema())
        result["parameters"]["required"] += ["semmem_update", "task_binding"]
    result["description"] += " In review-stmem wrap the action in action. A memory_query reads spatial history without executing motion."
    return result


class SpatialWorkerClient:
    def __init__(self, runtime):
        self.runtime, self.cfg = runtime, runtime.cfg["stmem"]
        self.directory = runtime.output / "stmem"
        self.directory.mkdir()
        from .stmem_frames import SceneFrames
        self.scene_frames=SceneFrames(self.directory/"scene_frames")
        self.latest_scene=None
        self.process = None
        self.lock = asyncio.Lock()
        self.last_context = {"schema": "oeh.stmem.context.v1", "entities": [], "status": "not_observed"}
        self.failed = False
        self.stderr = None
        self.calibration = json.loads(Path(self.cfg["calibration_file"]).read_text(encoding="utf-8"))
        if self.cfg["camera"] not in self.calibration:
            raise ValueError("stmem camera missing from calibration file")
        camera = self.calibration[self.cfg["camera"]]
        if not camera.get("frame_id") or camera.get("convention") != "opencv" or camera.get("units") != "m":
            raise ValueError("stmem calibration requires frame_id, convention=opencv, units=m")
        from .stmem_geometry import backproject
        if camera.get("source") != "observation":
            backproject(np.ones((4, 4)), np.ones((4, 4), bool), camera["K"], camera["world_from_camera"])

    async def start(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pop('semantic',None)
        cfg.pop('semmem',None)
        config = self.directory / "worker.json"
        config.write_text(json.dumps(cfg), encoding="utf-8")
        self.stderr = (self.directory / "worker.stderr").open("wb")
        self.process = await asyncio.create_subprocess_exec(self.cfg["python"], "-m", "oeh_policy.stmem_perception", "--config", str(config), "--output", str(self.directory / "perception"),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.stderr, start_new_session=True, limit=8*1024*1024)
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), self.cfg["startup_seconds"])
            if json.loads(line) != {"ready": True}:
                raise RuntimeError("perception worker failed preflight")
        except BaseException:
            await self.close()
            raise
        self.runtime.emit("stmem_ready", {"camera": self.cfg["camera"], "perception_seconds": self.cfg["perception_seconds"]})

    async def request(self, payload, timeout):
        async with self.lock:
            if self.failed or self.process is None or self.process.returncode is not None:
                raise RuntimeError("perception worker unavailable")
            rid = uuid.uuid4().hex
            self.process.stdin.write((json.dumps({"id": rid, **payload})+"\n").encode())
            try:
                await self.process.stdin.drain()
                response = json.loads(await asyncio.wait_for(self.process.stdout.readline(), timeout))
                if response.get("id") != rid:
                    raise RuntimeError("worker response mismatch")
            except BaseException:
                self.failed = True
                await self.close()
                raise
            if "error" in response:
                raise ValueError(response["error"])
            return response["result"]

    async def observe(self, packet_time):
        # Archive every native observation before worker inference, including chunk-internal frames.
        calibrations={k:v for k,v in self.calibration.items() if k=='cam_head' and v.get('source')!='observation'}
        calibrations.update(getattr(self.runtime,'stmem_camera_metadata',{}))
        self.latest_scene=self.scene_frames.capture(self.runtime.observation['vision'],
            native_step=self.runtime.gate.step,sim_time=packet_time,calibrations=calibrations,
            entity_ids=[e['entity_id'] for e in self.last_context.get('entities',[])])
        self.runtime.emit('scene_frame',{k:v for k,v in self.latest_scene.items() if k!='entity_ids'})
        camera = self.cfg["camera"]
        calibration = copy.deepcopy(self.calibration[camera])
        # Wrist transforms MUST be timestamped measurements supplied out-of-band by
        # the observation adapter. A static wrist extrinsic is never silently used.
        if camera != "cam_head" or calibration.get("source") == "observation":
            dynamic = getattr(self.runtime, "stmem_camera_metadata", {}).get(camera)
            if dynamic is None:
                raise RuntimeError("required same-observation camera calibration is missing")
            calibration.update(dynamic)
        path = self.directory / ("input-"+uuid.uuid4().hex+".npz")
        np.savez(path, rgb=self.runtime.observation["vision"][camera]["color"], K=calibration["K"], world_from_camera=calibration["world_from_camera"])
        try:
            result = await self.request({"operation": "observe", "input": str(path), "time": packet_time,
                                         "camera": camera, "frame_id": calibration["frame_id"],
                                         "scene_frame_id":self.latest_scene["frame_id"],"native_step":self.latest_scene["native_step"],
                                         "instruction":self.runtime.observation.get("instruction",self.runtime.observation.get("task_instruction",""))}, self.cfg["perception_seconds"])
            self.last_context = result["context"]
            self.scene_frames.link_entities(self.latest_scene["frame_id"],self.last_context)
            if "semantic_status" in result:
                self.last_context["semantic_status"] = result["semantic_status"]
            self.runtime.emit("stmem_observation", {"status": result["status"], "duration_s": result["duration_s"], "entities": len(self.last_context["entities"]), "semantic_status":result.get("semantic_status"), "semantic_calls":result.get("semantic_calls",[]),
                "description_states":self.last_context.get("description_states",{})})
            return self.last_context
        except (RuntimeError, ValueError, asyncio.TimeoutError, OSError) as exc:
            self.runtime.emit("stmem_unavailable", {"type": type(exc).__name__, "time": packet_time})
            self.last_context = {**self.last_context, "status": "stale_worker_unavailable", "requested_time": packet_time}
            return self.last_context
        finally:
            path.unlink(missing_ok=True)

    async def update(self,value,now,call_id):
        from .stmem_updates import validate_update
        validate_update(value)
        result=await self.request({"operation":"update","value":value,"time":now,"call_id":call_id},2.)
        self.last_context=await self.request({"operation":"context","time":now},2.)
        return result

    async def set_working_set(self,value,now,observation_id):
        result=await self.request({"operation":"working_set","value":value,"time":now,"observation_id":observation_id},2.)
        if result.get("status")!="stored":raise ValueError("STMem working-set persistence failed")
        self.last_context["working_set"]=result["working_set"]
        return result

    async def query(self, arguments, now):
        from .stmem_queries import validate_query,QueryInputError
        query = {k:v for k,v in arguments.items() if k!='kind'}
        try:
            validate_query(arguments)
        except QueryInputError as exc:
            return {"status":"invalid_query","field":exc.field,"error":str(exc),"no_execution":True,
                "hint":"Use exact entity_id for current/at/trajectory; resolve accepts one exact STMem label and returns IDs only; snapshot lists IDs"}
        if query['operation']=='frame_calibration':
            return self.scene_frames.calibration(**{k:v for k,v in query.items() if k!='operation'},now=now)
        if query['operation']=='frames':
            from .context_views import frames_query_model_view
            return frames_query_model_view(self.scene_frames.query(**{k:v for k,v in query.items() if k!='operation'},now=now))
        try:
            result = await self.request({"operation":"query","query":{**query,"now":now}},2.)
            self.last_context = await self.request({"operation":"context","time":now},2.)
            result["spatial_context"] = self.last_context
            if query['operation']=='snapshot' and query.get('include_images'):
                scenes=self.scene_frames.query(frame_ids=[self.latest_scene['frame_id']] if self.latest_scene and abs(self.latest_scene['sim_time']-now)<1e-6 else ['missing'],
                    k=4,limit=min(3,query.get('max_images',3)),now=now)
                result['_image_items']=scenes.pop('_image_items',[])
                from .context_views import frames_query_model_view
                result['scene_frames']=frames_query_model_view(scenes)
            return result
        except (RuntimeError,ValueError,asyncio.TimeoutError,OSError) as exc:
            status="invalid_query" if isinstance(exc,ValueError) and str(exc).startswith('QueryInputError:') else "memory_unavailable"
            return {"status":status,"error_type":type(exc).__name__,"error":str(exc)[:300],"query":query,
                "retryable":status=='memory_unavailable',"no_execution":True}

    async def close(self):
        try:
            if self.process and self.process.returncode is None:
                try:
                    if not self.failed:
                        try:
                            self.process.stdin.write((json.dumps({"id":uuid.uuid4().hex,"operation":"close"})+"\n").encode())
                            await self.process.stdin.drain()
                            await asyncio.wait_for(self.process.stdout.readline(),2.)
                        except (OSError,RuntimeError,asyncio.TimeoutError):
                            pass
                finally:
                    # Runtime's bounded cleanup can cancel the graceful wait; still reap this worker.
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await self.process.wait()
        finally:
            if self.stderr:
                self.stderr.close()
