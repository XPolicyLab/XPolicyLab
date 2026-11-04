"""MCP server for the speed-focused variant (stdio, or one local HTTP process shared by several client entries).

Same ControlRuntime contract as the migration (durable action IDs, measurement
provenance, release evidence), but: no estimate tool, compact JSON for the model,
full results archived unchanged, and per-call timing recorded for analysis.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Annotated, Any, Literal
import uuid

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictInt, create_model

from agp.experiments.control_runtime.mcp_server import ArmAction, GoalEvidence
from agp.experiments.control_runtime.runtime import ControlRuntime
from agp_core import compact, features, runner, runtime_ext
from agp_core.adapter import OursAdapter

Identifier = Annotated[str, Field(min_length=1, max_length=160)]


Pixel = Annotated[list[StrictInt], Field(min_length=2, max_length=2)]


class PointQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: Literal["left", "right"]
    cam: Literal["top", "left_wrist", "right_wrist"]
    u: Annotated[StrictInt, Field(ge=0, le=639)] | None = None
    v: Annotated[StrictInt, Field(ge=0, le=479)] | None = None
    target: Annotated[str, Field(min_length=1, max_length=160)]
    visibility: Annotated[str, Field(max_length=200)] | None = None
    plane_z: Annotated[float, Field(allow_inf_nan=False)] | None = None
    height_basis: Annotated[str, Field(max_length=200)] | None = None
    # candidates switch: several pixels, one kept by pick (solid | high | low) inside zr [zmin, zmax]
    cands: Annotated[list[Pixel], Field(min_length=1, max_length=8)] | None = None
    pick: Literal["solid", "high", "low"] | None = None
    zr: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    label: Annotated[str, Field(min_length=1, max_length=40)] | None = None


class PosePointQuery(PointQuery):
    """A measure query at one pixel; pose: true (top camera only) also returns the footprint of the object seen from above."""
    pose: bool | None = None


ArmActions = Annotated[dict[Literal["left", "right"], ArmAction], Field(min_length=1, max_length=2)]
# OURS_INTENT_MANIPULATE: the step tool's arm action with the seventh intent.
# Same class name and fields as ArmAction, so the listing differs only by "manipulate" in the intent enum; with the
# switch off the step tool keeps ArmAction (the listing is unchanged).
ManipulateArmAction = create_model("ArmAction", __base__=ArmAction, __module__=__name__, intent=(
    Literal["observe", "grasp", "transfer", "contact", "retreat", "place", "manipulate"], ...))
ManipulateArmActions = Annotated[dict[Literal["left", "right"], ManipulateArmAction], Field(min_length=1, max_length=2)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


class Responder:
    def __init__(self, session: Path, compact_views: bool = True):
        self.session = session
        self.timing = session / "tool_timing.jsonl"
        self.compact_views = compact_views

    def __call__(self, tool: str, started: float, full: dict, view: dict) -> CallToolResult:
        _write_json(self.session / "native_tool_results" / f"{time.monotonic_ns()}_{uuid.uuid4().hex}.json", full)
        if not self.compact_views:  # compact switch off: the runtime's full result, images still as image blocks
            view = {k: v for k, v in full.items() if k != "images"}
        text = json.dumps(view, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        blocks: list[TextContent | ImageContent] = [TextContent(type="text", text=text)]
        shown = []
        for reference in full.get("images", []):
            path = Path(reference["path"])
            path = (path if path.is_absolute() else self.session / path).resolve()
            if not path.is_relative_to(self.session):
                raise ValueError("image must be archived inside the session")
            raw = path.read_bytes()
            with Image.open(path) as decoded:
                if decoded.format != "PNG":
                    raise ValueError("inline images must be PNG")
                width, height = decoded.size
                decoded.verify()
            label = str(reference["label"])
            shown.append({"label": label, "sha256": hashlib.sha256(raw).hexdigest()})
            blocks.append(TextContent(type="text", text=f"Image {label} ({width}x{height})"))
            blocks.append(ImageContent(type="image", mimeType="image/png", data=base64.b64encode(raw).decode("ascii")))
        finished = time.time()
        with self.timing.open("a") as stream:
            stream.write(json.dumps({"tool": tool, "start": started, "end": finished,
                                     "duration_s": round(finished - started, 3), "ok": full.get("ok"),
                                     "text_chars": len(text), "images": [x["label"] for x in shown],
                                     "samples": (full.get("action") or {}).get("low_level_actions")}) + "\n")
        return CallToolResult(content=blocks, isError=full.get("ok") is False)


RGB_MEASURE_DOC = ("3D point at pixel (u,v) of one camera in the latest observation, expressed in the named arm's base "
                   "frame (m). The images have no depth: with plane_z at the table (<= 0.005) the pixel ray is "
                   "intersected with that horizontal plane; otherwise the height comes from matching the pixel in a "
                   "second camera (top <-> that arm's wrist), returned as z_from match or uncertain with a review "
                   "image, and a higher plane_z is only the fallback, used when the match is uncertain, missing or "
                   "more than 0.02 m away from it (z_from fallback). on_robot marks a pixel inside a robot arm's "
                   "outline. Returns measurement ids for step evidence.")
RGB_CANDIDATES_DOC = (" A query may give cands [[u,v],...] instead of u,v: all are measured and one is kept, the most "
                      "reliable (pick solid: a plane or confident match before a fallback or uncertain one, off the "
                      "robot) or the highest/lowest z (pick "
                      "high|low), after dropping points outside zr [zmin,zmax]. label names the kept point for later "
                      "run steps.")
DSL_MEASURE_DOC = " A query may also be a string '[label:] L|R top|lw|rw u,v[/u,v...] [z=..] [pick=..] [zr=a,b] target text'."
RUN_DOC = ("Execute up to 20 steps in order, server-side, stopping at the first surprise. Ids and observation threading "
           "are automatic; every action still passes the normal runtime checks. Step = {left?: arm, right?: arm, expect?: "
           "{arm: {gap_min_mm, gap_max_mm}}} or {measure: [queries], as: [labels]}. Arm = {do: ee|delta|open|close|home|"
           "hold|joints, intent?, to: [x,y,z] | {at: label_or_measurement_id, d: [dx,dy,dz]}, rot?: {w,x,y,z} | {down: deg} "
           "| {tilted: [az_deg, tilt_deg]} (default: keep current), mode?: linear|plan, speed?: precision|transit (default "
           "transit for observe/transfer/retreat intents), d: [dx,dy,dz] for delta, m?: measurement label or id, ev?: "
           "{...}, samples?}. show = {at: {\"<k>\": [labels]}, end: [labels], on_stop: [labels]}; evidence is shared by "
           "every arm action.")
RGB_DSL_RUN_DOC = (" A step may instead be a command string, e.g. 'L ee g+0,0,.05 down90 plan obs & R d 0,0,.03 "
                   "retreat', 'B close gap>2', 'L open support=flat surface below', or a measure step "
                   "'? g: L top 180,214 near object top'. show may be a string such as 'top lw 2:rw stop:top'.")
EV_DOC =" Retreat moves and home get a default clearance when none is given."
# Grip width (OURS_GRIP_WIDTH): the width action in the tool texts; with the switch off the step tool keeps its
# docstring and the run tool's text is unchanged.
STEP_GRIP_WIDTH_DOC = (
    "Execute one transaction for one or both arms on a shared clock; return compact outcome and fresh images. actions "
    "maps left/right to {command, args, intent, measurement_id?, evidence?}. Commands: move_ee(position{x,y,z} or "
    "position_offset + measurement_id, rotation{w,x,y,z}, mode plan|linear, speed_profile precision|transit), "
    "move_delta(dpos[3] m, optional drot_deg[3], speed_profile), move_joints(joints[6]), home, gripper(action "
    "open|close, or action width with width_mm: the pad gap in mm, 0.8 closed .. 88.8 fully open), hold(samples "
    "1..25). Motion options: max_samples, settle_samples 0..10, position_tolerance_m, rotation_tolerance_rad.")
GRIP_WIDTH_RUN_DOC = (" do: width with width_mm (the pad gap in mm, 0.8 closed .. 88.8 fully open) moves the fingers to "
                      "that gap, e.g. 'L width 30'.")
# Pose query (OURS_POSE_QUERY): the pose option in the tool texts; with the switch off the texts and the measure
# tool's query model are unchanged.
POSE_MEASURE_DOC = (" pose: true (string option `pose`, top camera only) also returns the footprint of the object at that "
                    "pixel seen from above: xy centroid, axis_deg / tail_deg heading, edge_deg, length_mm, width_mm, status.")
POSE_RUN_DOC = " A measure query may add `pose` (e.g. '? t: L top 180,214 z=.015 pose object top') for the object's footprint."


def create_server(session: Path, runtime: ControlRuntime, port: int = 8000) -> FastMCP:
    session = session.resolve()
    goals = GoalEvidence(session, session / "goal")
    flags = features.current()
    respond = Responder(session, compact_views=flags["compact"])
    labels: dict[str, str] = {}  # measurement labels persist for the episode
    measure_doc = (RGB_MEASURE_DOC + (RGB_CANDIDATES_DOC if flags["candidates"] else "")
                   + (DSL_MEASURE_DOC if flags["dsl"] else ""))
    run_doc = (RUN_DOC + (RGB_DSL_RUN_DOC if flags["dsl"] else "") + (EV_DOC if flags["ev_defaults"] else ""))
    if flags.get("grip_width"):                                                                # the width action
        run_doc += GRIP_WIDTH_RUN_DOC
    if flags.get("pose_query"):                                                              # the pose option
        measure_doc += POSE_MEASURE_DOC
        run_doc += POSE_RUN_DOC
    query_model = PosePointQuery if flags.get("pose_query") else PointQuery
    step_tool = {"description": STEP_GRIP_WIDTH_DOC} if flags.get("grip_width") else {}
    step_actions = ManipulateArmActions if flags.get("intent_manipulate") else ArmActions     # OURS_INTENT_MANIPULATE
    # Tools are synchronous, so FastMCP runs them on the event loop one at a time: two client
    # entries (direct calls and code-mode calls) can share this one runtime safely.
    server = FastMCP("astra_control", log_level="WARNING", host="127.0.0.1", port=port, instructions=(
        "Each observe/step result starts with one compact JSON text block, followed by labelled "
        "top, left_wrist and right_wrist images. Never retry an uncertain action with a new action_id; "
        "use action_status."))

    @server.tool(structured_output=False)
    def goal() -> CallToolResult:
        """Return the task instruction."""
        started = time.time()
        full = goals.read([], None)
        return respond("goal", started, full, compact.generic(full))

    @server.tool(structured_output=False)
    def observe() -> CallToolResult:
        """Current observation: compact state of both arms and top, left_wrist, right_wrist images. No physics advances."""
        started = time.time()
        full = runtime.observe()
        view = compact.observation(full) or {}
        view["ok"] = full.get("ok")
        return respond("observe", started, full, view)

    @server.tool(structured_output=False, description=measure_doc)
    def measure(observation_id: Identifier,
                queries: Annotated[list[query_model | str], Field(min_length=1, max_length=8)]) -> CallToolResult:
        started = time.time()
        raw = [q if isinstance(q, str) else q.model_dump(exclude_none=True) for q in queries]
        try:
            full, points, ok, new, error = runner.measure_queries(runtime, observation_id, raw, None, labels, flags)
        except (KeyError, ValueError, TypeError) as exc:
            full = {"ok": False, "error_code": "INVALID_MEASUREMENT", "error": str(exc)}
            return respond("measure", started, full, compact.measure(full))
        view = {"ok": ok, "observation_id": observation_id, "points": points}
        if error:
            view["error"] = error
        if new:
            view["labels"] = new
        return respond("measure", started, dict(full, ok=ok), view)

    @server.tool(structured_output=False, **step_tool)
    def step(action_id: Identifier, observation_id: Identifier, actions: step_actions) -> CallToolResult:
        """Execute one transaction for one or both arms on a shared clock; return compact outcome and fresh images. actions maps left/right to {command, args, intent, measurement_id?, evidence?}. Commands: move_ee(position{x,y,z} or position_offset + measurement_id, rotation{w,x,y,z}, mode plan|linear, speed_profile precision|transit), move_delta(dpos[3] m, optional drot_deg[3], speed_profile), move_joints(joints[6]), home, gripper(action open|close), hold(samples 1..25). Motion options: max_samples, settle_samples 0..10, position_tolerance_m, rotation_tolerance_rad."""
        started = time.time()
        specs = {}
        for arm, spec in actions.items():
            item = spec.model_dump(exclude_none=True)
            evidence = runner.default_evidence(item.get("command"), item.get("intent"), item.get("evidence") or {}, flags)
            if evidence:
                item["evidence"] = evidence
            specs[arm] = item
        full = runtime.step(action_id=action_id, observation_id=observation_id, actions=specs)
        return respond("step", started, full, compact.step(full))

    if flags["run"]:  # run switch off: no server-side chaining, the policy chains `step` itself
        @server.tool(structured_output=False, description=run_doc)
        def run(observation_id: Identifier,
                steps: Annotated[list[dict[str, Any] | str], Field(min_length=1, max_length=20)],
                show: dict[str, Any] | list[str] | str | None = None,
                evidence: dict[str, Any] | None = None) -> CallToolResult:
            started = time.time()
            full, view = runner.execute(runtime, observation_id, steps, show, evidence, labels, flags)
            return respond("run", started, full, view)

    @server.tool(structured_output=False)
    def action_status(action_id: Identifier) -> CallToolResult:
        """Reconcile an existing action after an uncertain result. Never dispatches motion."""
        started = time.time()
        full = runtime.action_status(action_id=action_id)
        view = compact.step(full) if "action" in full else compact.generic(full)
        return respond("action_status", started, full, view)

    @server.tool(structured_output=False)
    def assess(observation_id: Identifier, outcome: Literal["success", "failure", "unknown"],
               evidence: dict[str, Any]) -> CallToolResult:
        """Record your final task judgement with evidence.summary. The benchmark evaluator decides independently."""
        started = time.time()
        full = runtime.assess(observation_id=observation_id, outcome=outcome, evidence=evidence)
        return respond("assess", started, full, compact.generic(full))

    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--http", type=int, help="serve streamable HTTP on 127.0.0.1:PORT/mcp instead of stdio")
    options = parser.parse_args()
    session = options.session.resolve()
    # ControlRuntime itself unless OURS_GUARD_V2 and OURS_ARM_GUARD are on (agp_core/runtime_ext.py); with
    # OURS_INTENT_MANIPULATE on, that class with the intent manipulate; the keep-unstarted class also when
    # OURS_KEEP_UNSTARTED is on
    runtime = runtime_ext.runtime_class()(session, OursAdapter(session, timeout_s=900))
    try:
        if options.http:
            create_server(session, runtime, options.http).run(transport="streamable-http")
        else:
            create_server(session, runtime).run(transport="stdio")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
