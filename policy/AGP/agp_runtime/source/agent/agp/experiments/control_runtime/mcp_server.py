"""STDIO MCP boundary for the native robot runtime.

Images are returned inline in the same result as their observation or action.
The explicit fixture mode imports only a data-only test backend.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Annotated, Any, Literal
import uuid

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

Identifier = Annotated[str, Field(min_length=1, max_length=160)]
FiniteSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Command = Literal["move_ee", "move_delta", "move_joints", "home", "gripper", "hold"]
Intent = Literal["observe", "grasp", "transfer", "contact", "retreat", "place"]


class MeasureQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Annotated[str, Field(min_length=1, max_length=160)]
    cam: Annotated[str, Field(min_length=1, max_length=40)]
    u: Annotated[StrictInt, Field(ge=0)]
    v: Annotated[StrictInt, Field(ge=0)]
    plane_z: Annotated[float, Field(allow_inf_nan=False)] | None = None
    height_basis: Annotated[str, Field(max_length=300)] | None = None
    visibility: Annotated[str, Field(max_length=300)] | None = None


class ArmMeasureQuery(MeasureQuery):
    arm: Literal["left", "right"]
    cam: Literal["top", "left_wrist", "right_wrist"]


class ArmAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Command
    args: dict[str, Any]
    intent: Intent
    measurement_id: Identifier | None = None
    evidence: dict[str, Any] | None = None


ArmActions = Annotated[dict[Literal["left", "right"], ArmAction], Field(min_length=1, max_length=2)]


class EstimateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Command
    args: dict[str, Any]
    measurement_id: Identifier | None = None


EstimateStage = Annotated[dict[Literal["left", "right"], EstimateAction], Field(min_length=1, max_length=2)]


class PreviewRequest(BaseModel):
    """Caller-selected stage-0 samples at a single captured viewpoint."""
    model_config = ConfigDict(extra="forbid")
    camera: Literal["top", "left_wrist", "right_wrist"]
    samples: Annotated[list[Annotated[StrictInt, Field(ge=0)]], Field(min_length=1, max_length=3)]

    @field_validator("samples")
    @classmethod
    def unique_samples(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("Preview samples must be unique")
        return sorted(values)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _inline_result(session: Path, result: dict[str, Any]) -> CallToolResult:
    """Archive exact metadata and transmit original, validated PNG bytes."""
    archive = session / "native_tool_results" / f"{time.monotonic_ns()}_{uuid.uuid4().hex}.json"
    _write_json(archive, result)
    summary = {key: value for key, value in result.items() if key != "images"}
    attachments = []
    blocks: list[TextContent | ImageContent] = []
    for reference in result.get("images", []):
        path = Path(reference["path"])
        path = (path if path.is_absolute() else session / path).resolve()
        if not path.is_relative_to(session):
            raise ValueError("image must be archived inside the session")
        raw = path.read_bytes()
        with Image.open(path) as decoded:
            if decoded.format != "PNG":
                raise ValueError("inline images must be original PNG files")
            width, height = decoded.size
            decoded.verify()
        label = str(reference["label"])
        attachments.append({"label": label, "width": width, "height": height,
                            "sha256": hashlib.sha256(raw).hexdigest()})
        kind = reference.get("kind")
        if kind == "derived_hypothetical_robot_projection":
            caption = "derived hypothetical robot projection; exact PNG bytes"
        elif kind == "captured_rgb":
            caption = "original captured RGB; exact PNG bytes"
        else:
            caption = "Original PNG"
        blocks.extend([
            TextContent(type="text", text=f"Image {label}. {caption}, {width} x {height} pixels."),
            ImageContent(type="image", mimeType="image/png", data=base64.b64encode(raw).decode("ascii")),
        ])
    if attachments:
        summary["attached_images"] = attachments
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(summary, ensure_ascii=False, allow_nan=False)), *blocks],
        isError=result.get("ok") is False,
    )


class GoalEvidence:
    """A fixed per-session copy of configured goal assets, with bounded decoding."""

    def __init__(self, session: Path, source: Path):
        self.directory = session / "native_goal"
        self.instruction = ""
        self.references: list[Path] = []
        self.videos: dict[str, Path] = {}
        self.video_durations: dict[str, float] = {}
        if not source.exists():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        references = sorted(source.glob("*.png"))
        if len(references) > 4:
            raise ValueError("goal directory must contain at most four reference PNGs")
        instructions = [source / "INSTRUCTION.md"] if (source / "INSTRUCTION.md").is_file() else []
        for original in references + sorted(source.glob("*.mp4")) + instructions:
            destination = self.directory / original.name
            if original.resolve() != destination.resolve():
                if destination.exists() and destination.read_bytes() != original.read_bytes():
                    raise ValueError("goal assets changed after this session began")
                if not destination.exists():
                    shutil.copy2(original, destination)
            if destination.name == "INSTRUCTION.md":
                self.instruction = destination.read_text(encoding="utf-8").strip()
            elif destination.suffix == ".png":
                self.references.append(destination)
            else:
                self.videos[destination.name] = destination

    def read(self, timestamps: list[float], video: str | None) -> dict[str, Any]:
        images = [{"label": f"goal reference {path.name}", "path": str(path)} for path in self.references]
        for name, path in self.videos.items():
            if name not in self.video_durations:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                    check=True, capture_output=True, text=True, timeout=15,
                )
                self.video_durations[name] = float(json.loads(probe.stdout)["format"]["duration"])
        if timestamps:
            if video is None:
                if len(self.videos) != 1:
                    raise ValueError("select a video from available_videos")
                video = next(iter(self.videos))
            if video not in self.videos:
                raise ValueError("video must name one of the configured goal videos")
            path = self.videos[video]
            duration = self.video_durations[video]
            if any(timestamp >= duration for timestamp in timestamps):
                raise ValueError(f"demo timestamp must be less than video duration {duration:.6f} seconds")
            for timestamp in timestamps:
                key = hashlib.sha256(f"{video}\n{timestamp.hex()}".encode()).hexdigest()[:20]
                destination = self.directory / "frames" / f"{key}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
                         "-i", str(path), "-frames:v", "1", "-y", str(destination)],
                        check=True, capture_output=True, timeout=30,
                    )
                images.append({"label": f"goal demo {video} at requested {timestamp:g} seconds", "path": str(destination)})
        if not images and not self.videos and not self.instruction:
            raise ValueError("no goal assets configured")
        return {"ok": True, "instruction": self.instruction, "available_videos": [
            {"name": name, "duration_s": self.video_durations[name]} for name in self.videos
        ], "demo_timestamps_s": timestamps, "images": images}


def create_server(session: Path, runtime: Any, goal_dir: Path | None = None) -> FastMCP:
    session = session.resolve()
    goals = GoalEvidence(session, goal_dir or session / "goal")
    server = FastMCP(
        "astra_control",
        instructions="Use returned images directly. Each step returns its completion and fresh camera images in one result. Never retry an uncertain action with a new action_id. Reconcile with action_status.",
        log_level="WARNING",
    )

    @server.tool(structured_output=False)
    def observe() -> CallToolResult:
        """Get one coherent current observation with all configured camera images (simulation: top, left_wrist, right_wrist) and measured arm states. Observation does not advance simulation physics."""
        return _inline_result(session, runtime.observe())

    def measure(observation_id: Identifier, queries: Annotated[list[MeasureQuery], Field(min_length=1, max_length=8)]) -> CallToolResult:
        """Measure selected visible pixels in the latest observation. Geometry does not establish object identity. Coordinates are original image pixels; optional plane_z is metres."""
        return _inline_result(session, runtime.measure(
            observation_id=observation_id, queries=[query.model_dump(exclude_none=True) for query in queries],
        ))

    def step(action_id: Identifier, observation_id: Identifier, command: Command,
             args: dict[str, Any], intent: Intent, measurement_id: Identifier | None = None,
             evidence: dict[str, Any] | None = None) -> CallToolResult:
        """Execute one action and return fresh images. move_ee and move_delta accept speed_profile precision or transit. Transit requires observe, transfer or retreat intent and evidence.clearance for the entire path. Keep grasp, contact, short trial lifts and near-obstacle motion at precision. home uses the observation-posture transit profile and requires clearance evidence. ok means execution and observation completed, not task success. action.motion_outcome=held_short and target_reached=false require visual evaluation of seating versus obstruction. Use returned images directly. Release can reconfirm lost target evidence with a measurement_id from the latest observation plus target_unchanged, engagement and support, without dummy motion. action_id is unique and durable. Args use the existing bridge units. Transfer and initial contact require a valid measurement_id and target_unchanged evidence. Immediately after successful contact, move_delta may omit measurement_id for a nonzero translation of at most 20 mm without rotation, using that contact observation and evidence.previous_action_id, target_unchanged and adjustment_reason. Failure, held_short or intervening actions require remeasurement for further contact. Release must satisfy the session's drop, support or engagement_and_support requirement. For drop, provide landing evidence describing the destination and clear fall path in the latest images. To open a visibly empty gripper in place, use release_requirement=empty and empty evidence from the latest images. After any grasp attempt also provide object_location locating the object separately and opening_clearance for the fingers. Unknown or occluded possession requires another observation. Unknown actions must be reconciled, never repeated."""
        return _inline_result(session, runtime.step(
            action_id=action_id, observation_id=observation_id, command=command, args=args,
            intent=intent, measurement_id=measurement_id, evidence=evidence,
        ))

    if getattr(runtime, "dual_arm", False):
        def measure_arms(observation_id: Identifier,
                         queries: Annotated[list[ArmMeasureQuery], Field(min_length=1, max_length=8)]) -> CallToolResult:
            """Measure visible pixels from top, left_wrist or right_wrist. Each query MUST name the target arm left/right; camera and arm are independent. Output is in that arm's base frame. Geometry does not establish object identity. Optional plane_z is metres in the selected arm base and requires height_basis. Evidence and cache entries belong to that arm only."""
            return _inline_result(session, runtime.measure(observation_id=observation_id,
                queries=[query.model_dump(exclude_none=True) for query in queries]))

        def step_arms(action_id: Identifier, observation_id: Identifier, actions: ArmActions) -> CallToolResult:
            """Execute one durable transaction with one or both arms, returning fresh top and both wrist images. actions maps left/right to command,args,intent,measurement_id,evidence. Commands: move_ee(position or measurement position_offset xyz, rotation wxyz, mode linear/plan), move_delta(dpos metres,drot_deg), move_joints(joints radians), home, gripper(action open/close), hold(samples integer 1..25). hold advances counted physical time at measured joints; it does not establish a post-grasp lift. Motion commands accept max_samples 1..1000 (execute a prefix checkpoint and discard suffix), settle_samples 0..10 (default 3 counted endpoint holds), position_tolerance_m .0005..01 (default .002), rotation_tolerance_rad .005..1 (default .02). Cartesian moves accept speed_profile precision/transit; transit requires observe/transfer/retreat intent and evidence.clearance. home requires clearance. One or both arms advance on a shared sample clock; unequal trajectories hold endpoints. No verified inter-arm swept collision avoidance: inspect both arms and clearances. Check action.arms for each arm's achieved pose, target_reached and motion_outcome; shared ok is not task success. Short/checkpoint/failed motion requires reobservation and measured correction before dependent grasp or transfer. Measurements MUST belong to the action arm. Transfer/contact require measurement_id and target_unchanged. A relative contact continuation may reference only the latest successful contact on the SAME arm, using previous_action_id,target_unchanged,adjustment_reason and a nonzero <=20mm translation without rotation. Close records attempted grasp, never possession; visually verify a short precision trial lift. Opening requires release_requirement matching the session and support/engagement or landing evidence. Recovery release uses release_purpose=recovery with support. Empty opening requires empty evidence and, after a grasp attempt, object_location and opening_clearance. Reconfirm engagement release with a latest same-arm measurement and target_unchanged if provenance was invalidated. Any uncertain pair blocks BOTH arms: reconcile action_status, never retry with a new ID."""
            return _inline_result(session, runtime.step(action_id=action_id, observation_id=observation_id,
                actions={arm: spec.model_dump(exclude_none=True) for arm, spec in actions.items()}))

        server.tool(name="measure", structured_output=False)(measure_arms)
        server.tool(name="step", structured_output=False)(step_arms)
    else:
        server.tool(name="measure", structured_output=False)(measure)
        server.tool(name="step", structured_output=False)(step)

    if getattr(runtime, "supports_estimate", False) is True:
        @server.tool(structured_output=False)
        def estimate(observation_id: Identifier,
                     stages: Annotated[list[EstimateStage], Field(min_length=1, max_length=12)],
                     preview: PreviewRequest | None = None) -> CallToolResult:
            """Estimate ordered candidate stages without physical steps or grasp-state changes. Each stage may command one or both arms on the shared clock. Later endpoints are conditional predictions, not observations or proof of possession, reachability or task success. Budget excess returns estimated costs without executing. Real steps still require fresh evidence and independent planning. Optional preview projects both robots at 1..3 caller-selected sample indices of stage 0 onto one original captured camera view. Sample 0 is the captured start; 1..N are nominal shared samples (get N from an ordinary estimate). No stage-0 gripper operation is supported. Images are hypothetical robot geometry, including hidden geometry, at a fixed captured viewpoint; they do not establish collision freedom, future visibility, possession or execution binding. Steps independently replan."""
            resolved = [{arm: spec.model_dump(exclude_none=True) for arm, spec in stage.items()}
                        for stage in stages]
            if preview is None:
                return _inline_result(session, runtime.estimate(observation_id, resolved))
            return _inline_result(session, runtime.estimate(observation_id, resolved,
                preview=preview.model_dump()))

    @server.tool(structured_output=False)
    def action_status(action_id: Identifier) -> CallToolResult:
        """Reconcile an existing action with its backend response. This never dispatches or retries motion."""
        return _inline_result(session, runtime.action_status(action_id=action_id))

    @server.tool(structured_output=False)
    def assess(observation_id: Identifier, outcome: Annotated[str, Field(min_length=1, max_length=80)],
               evidence: dict[str, Any]) -> CallToolResult:
        """Record your task judgement against the latest image evidence. Independent outcome verification remains pending."""
        return _inline_result(session, runtime.assess(observation_id=observation_id, outcome=outcome, evidence=evidence))

    @server.tool(structured_output=False)
    def goal(demo_timestamps_s: Annotated[list[FiniteSeconds], Field(max_length=4)] = [],
             video: Annotated[str, Field(min_length=1, max_length=200)] | None = None) -> CallToolResult:
        """Return archived task instructions, reference PNGs and up to four selected demo frames inline. Timestamps are seconds into the demonstration. Uses configured goal assets and cannot move the robot."""
        return _inline_result(session, goals.read(demo_timestamps_s, video))

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--session", type=Path, help="Existing real robot session")
    modes.add_argument("--fixture-session", type=Path, help="Explicit offline fixture session; cannot load the robot adapter")
    parser.add_argument("--goal-dir", type=Path)
    options = parser.parse_args()
    session = (options.fixture_session or options.session).resolve()
    if not session.is_dir():
        parser.error("session directory must already exist")
    if options.fixture_session:
        spec = importlib.util.spec_from_file_location("astra_mcp_offline_fixture", Path(__file__).parent / "tests" / "test_mcp.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime = module.FixtureRuntime(session)
    else:
        from adapter import RobotAdapter
        from runtime import ControlRuntime
        runtime = ControlRuntime(session, RobotAdapter(session))
    try:
        create_server(session, runtime, options.goal_dir).run(transport="stdio")
    finally:
        close = getattr(runtime, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
