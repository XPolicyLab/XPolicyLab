"""Offline web viewer for L4 Inspect decisions aligned with RoboDojo videos."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from .trace import SCHEMA_VERSION


CAMERAS = ("head", "left_wrist", "right_wrist")


def probe_video(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=avg_frame_rate,nb_read_frames,nb_frames,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(process.stdout)["streams"][0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
    fps = float(numerator) / float(denominator)
    frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": float(stream.get("duration") or (frame_count / fps)),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def _find_videos(video_dir: Path, episode_index: int | None = None) -> dict[str, Path]:
    prefix = f"episode_{episode_index:07d}_" if episode_index is not None else "*"
    videos = {}
    for camera in CAMERAS:
        matches = sorted(video_dir.glob(f"{prefix}cam_{camera}_*.mp4"))
        if matches:
            videos[camera] = matches[-1]
    if not videos:
        raise FileNotFoundError(f"No RoboDojo camera videos found under {video_dir}")
    return videos


def _load_audit(trace_dir: Path) -> dict[str, Any]:
    path = trace_dir / "l4_inspect_transcript.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing L4 Inspect transcript: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    version = audit.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported L4 Inspect trace schema {version!r}; expected {SCHEMA_VERSION!r}"
        )
    turns = audit.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("L4 Inspect trace has no structured turns")
    return audit


def _official_result(
    audit: dict[str, Any],
    video_dir: Path,
    episode_index: int | None,
) -> tuple[bool | None, float | None]:
    result_path = video_dir / "_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        details = result.get("details", {})
        detail = (
            details.get(str(episode_index), {})
            if episode_index is not None and isinstance(details, dict)
            else next(iter(details.values()), {})
            if isinstance(details, dict)
            else {}
        )
        if isinstance(detail, dict) and "success" in detail:
            return bool(detail["success"]), detail.get("score", result.get("score"))
    success = audit.get("official_success")
    if isinstance(success, list) and success:
        return bool(success[0]), None
    return None, None


def _accepted_call(turn: dict[str, Any]) -> dict[str, Any]:
    calls = turn.get("llm_calls") or []
    return next(
        (call for call in reversed(calls) if call.get("accepted")),
        calls[-1] if calls else {},
    )


def build_manifest(
    trace_dir: Path,
    video_dir: Path,
    *,
    probe: Callable[[Path], dict[str, Any]] = probe_video,
    episode_index: int | None = None,
    video_url_prefix: str = "/video",
) -> dict[str, Any]:
    """Merge a v1 L4 trace with camera metadata for browser playback."""
    trace_dir = trace_dir.resolve()
    video_dir = video_dir.resolve()
    audit = _load_audit(trace_dir)
    video_paths = _find_videos(video_dir, episode_index)
    videos = {
        camera: {
            "url": f"{video_url_prefix}/{camera}",
            "name": path.name,
            **probe(path),
        }
        for camera, path in video_paths.items()
    }
    warnings = [
        f"missing {camera} camera video"
        for camera in CAMERAS
        if camera not in videos
    ]
    raw_turns = audit["turns"]
    turns = []
    for index, raw in enumerate(raw_turns):
        observation = raw.get("observation") or {}
        observation_frames = observation.get("cameras") or {}
        call = _accepted_call(raw)
        decision = raw.get("decision") or {}
        execution = raw.get("execution")
        tool = decision.get("tool") or call.get("tool") or "error"
        playback: dict[str, dict[str, int]] = {}
        for camera, info in videos.items():
            bounds = observation_frames.get(camera) or {}
            frame = bounds.get("frame")
            if frame is None:
                warnings.append(
                    f"policy step {raw.get('policy_step', index)} has no {camera} observation frame"
                )
                frame = 0
            start = max(0, int(frame))
            next_frame = None
            if index + 1 < len(raw_turns):
                next_frame = (
                    ((raw_turns[index + 1].get("observation") or {}).get("cameras") or {})
                    .get(camera, {})
                    .get("frame")
                )
            end = int(next_frame) if next_frame is not None else int(info["frame_count"])
            end = max(start + 1, end)
            if start >= int(info["frame_count"]):
                warnings.append(
                    f"{camera} observation frame {start} is outside {info['frame_count']} frames"
                )
                start = max(0, int(info["frame_count"]) - 1)
            playback[camera] = {
                "start": start,
                "end": min(max(start + 1, end), int(info["frame_count"])),
            }
        next_measured_state = (
            (raw_turns[index + 1].get("observation") or {}).get("state")
            if index + 1 < len(raw_turns)
            else None
        )
        turns.append(
            {
                "policy_step": raw.get("policy_step", index),
                "tool": tool,
                "arguments": call.get("arguments"),
                "llm_calls": raw.get("llm_calls") or [],
                "decision": decision,
                "execution": execution,
                "observation_state": observation.get("state") or {},
                "observation_frames": observation_frames,
                "next_measured_state": next_measured_state,
                "playback": playback,
                "error": raw.get("error"),
            }
        )
    success, score = _official_result(audit, video_dir, episode_index)
    return {
        "trace_dir": str(trace_dir),
        "video_dir": str(video_dir),
        "episode": {
            "task": audit.get("task"),
            "run_id": audit.get("run_id"),
            "layout_id": audit.get("layout_id"),
            "instruction": audit.get("instruction"),
            "termination_reason": audit.get("termination_reason"),
            "failure_kind": audit.get("failure_kind"),
            "error_message": audit.get("error_message"),
            "official_success": success,
            "score": score,
            "llm_calls": audit.get("llm_calls"),
        },
        "videos": videos,
        "turns": turns,
        "warnings": warnings,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L4 Inspect Timeline</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif;height:100vh;overflow:hidden}
header{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
.summary{display:flex;align-items:center;gap:10px}.result{font-weight:800;padding:4px 9px;border:1px solid var(--line);border-radius:999px}.success{color:#aff5b4}.failure{color:#ffb4ad}.unknown{color:#e3b341}
#instruction{margin-top:7px;color:#fff8c5}#warnings{margin-top:5px;color:#f2cc60}
#layout{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 95px)}aside{overflow:auto;border-right:1px solid var(--line);background:var(--panel)}
.turn{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}.turn:hover,.turn.active{background:#21262d}.turn small{display:block;color:var(--muted)}
main{overflow:auto;padding:14px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.camera{background:#000;border:1px solid var(--line);border-radius:6px;overflow:hidden}.camera h3{font-size:12px;margin:0;padding:6px 9px;background:var(--panel)}video{display:block;width:100%}
.controls{display:flex;align-items:center;gap:8px;margin:12px 0}.controls input{flex:1}.badge{padding:2px 7px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
.details{display:grid;grid-template-columns:1fr 1fr;gap:10px}.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;min-width:0}.wide{grid-column:1/-1}.card h3{margin:0 0 8px;font-size:13px}pre{white-space:pre-wrap;word-break:break-word;margin:0}
@media(max-width:1000px){.videos,.details{grid-template-columns:1fr}#layout{grid-template-columns:220px 1fr}}
</style>
</head>
<body>
<header><div class="summary"><span id="result" class="result unknown">RESULT UNKNOWN</span><span id="episode"></span></div><div id="instruction"></div><div id="warnings"></div></header>
<div id="layout"><aside id="turns"></aside><main>
<div class="videos" id="videos"></div>
<div class="controls"><button id="prev">◀ frame</button><button id="play-turn">Replay Decision</button><button id="play-all">Play Full Episode</button><button id="pause">Pause</button><button id="next">frame ▶</button><input id="scrub" type="range" min="0" max="1" step="1"><span class="badge" id="position"></span></div>
<div class="details">
<section class="card"><h3>move_joints / stop decision</h3><pre id="decision"></pre></section>
<section class="card"><h3>Execution</h3><pre id="execution"></pre></section>
<section class="card"><h3>Measured joint state before</h3><pre id="before"></pre></section>
<section class="card"><h3>Measured joint state at next observation</h3><pre id="after"></pre></section>
<section class="card wide"><h3>LLM calls and repairs</h3><pre id="calls"></pre></section>
</div></main></div>
<script>
let manifest,selected=0,playing=false,stopFrame=0,raf=null;
const videoEls={};
const clamp=(v,l,h)=>Math.min(Math.max(v,l),h);
const head=()=>videoEls.head||Object.values(videoEls)[0];
const fps=cam=>(manifest.videos[cam]||Object.values(manifest.videos)[0]).fps;
const bounds=(turn,cam='head')=>turn.playback[cam]||Object.values(turn.playback)[0]||{start:0,end:1};
const frame=()=>head()?Math.round(head().currentTime*fps('head')):0;
function turnAt(value){for(let i=manifest.turns.length-1;i>=0;i--)if(value>=bounds(manifest.turns[i]).start)return i;return 0}
function cameraFrame(value,cam){if(cam==='head'||videoEls[cam]===head())return value;const turn=manifest.turns[turnAt(value)],a=bounds(turn),b=bounds(turn,cam),p=clamp((value-a.start)/Math.max(1,a.end-a.start),0,1);return clamp(b.start+Math.round(p*Math.max(1,b.end-b.start)),b.start,Math.max(b.start,b.end-1))}
function render(id,value){document.querySelector(id).textContent=JSON.stringify(value,null,2)}
function select(index){selected=clamp(index,0,manifest.turns.length-1);document.querySelectorAll('.turn').forEach((e,i)=>e.classList.toggle('active',i===selected));const turn=manifest.turns[selected];render('#decision',{policy_step:turn.policy_step,tool:turn.tool,arguments:turn.arguments,...turn.decision,error:turn.error});render('#execution',turn.execution);render('#before',turn.observation_state);render('#after',turn.next_measured_state);render('#calls',turn.llm_calls)}
function seek(value){value=clamp(Math.round(value),0,+document.querySelector('#scrub').max);for(const [cam,video] of Object.entries(videoEls))video.currentTime=cameraFrame(value,cam)/fps(cam);document.querySelector('#scrub').value=value;const index=turnAt(value);if(index!==selected)select(index);position()}
function position(){document.querySelector('#position').textContent=`frame ${frame()} · decision ${selected+1}/${manifest.turns.length} · ${manifest.turns[selected].tool}`}
function pause(){playing=false;if(raf!==null)cancelAnimationFrame(raf);raf=null;Object.values(videoEls).forEach(v=>v.pause())}
function tick(){const value=frame();position();for(const [cam,video] of Object.entries(videoEls))if(video!==head()){const target=cameraFrame(value,cam)/fps(cam);if(Math.abs(video.currentTime-target)>.08)video.currentTime=target}if(value>=stopFrame){pause();seek(stopFrame);return}if(playing)raf=requestAnimationFrame(tick)}
async function play(full){pause();const turn=manifest.turns[selected];const start=full?0:bounds(turn).start;stopFrame=full?(manifest.videos.head||Object.values(manifest.videos)[0]).frame_count-1:bounds(turn).end-1;seek(start);playing=true;await Promise.allSettled(Object.values(videoEls).map(v=>v.play()));raf=requestAnimationFrame(tick)}
async function init(){manifest=await fetch('api/manifest').then(r=>{if(!r.ok)throw new Error(`manifest ${r.status}`);return r.json()});const success=manifest.episode.official_success,result=document.querySelector('#result');result.textContent=success===true?'OFFICIAL SUCCESS':success===false?'OFFICIAL FAILURE':'RESULT UNKNOWN';result.className=`result ${success===true?'success':success===false?'failure':'unknown'}`;document.querySelector('#episode').textContent=`${manifest.episode.task||''} · layout ${manifest.episode.layout_id??'?'} · ${manifest.episode.llm_calls??'?'} LLM calls · ${manifest.episode.termination_reason||'running'}`;document.querySelector('#instruction').textContent=manifest.episode.instruction||'';document.querySelector('#warnings').textContent=manifest.warnings.join(' · ');for(const [cam,info] of Object.entries(manifest.videos)){const box=document.createElement('div');box.className='camera';box.innerHTML=`<h3>${cam} · ${info.frame_count} frames · ${info.fps.toFixed(2)} fps</h3><video preload="auto" playsinline muted src="${info.url}"></video>`;document.querySelector('#videos').appendChild(box);videoEls[cam]=box.querySelector('video')}manifest.turns.forEach((turn,i)=>{const b=bounds(turn),row=document.createElement('div');row.className='turn';row.innerHTML=`<b>${i+1}. ${turn.tool}</b><small>observation frame ${b.start} · env ${turn.execution?.env_step_start??'?'}→${turn.execution?.env_step_end??'?'}</small>`;row.onclick=()=>{select(i);seek(b.start)};document.querySelector('#turns').appendChild(row)});document.querySelector('#scrub').max=Math.max(0,(manifest.videos.head||Object.values(manifest.videos)[0]).frame_count-1);select(0);seek(bounds(manifest.turns[0]).start)}
document.querySelector('#prev').onclick=()=>{pause();seek(frame()-1)};document.querySelector('#next').onclick=()=>{pause();seek(frame()+1)};document.querySelector('#play-turn').onclick=()=>play(false);document.querySelector('#play-all').onclick=()=>play(true);document.querySelector('#pause').onclick=pause;document.querySelector('#scrub').oninput=e=>{pause();seek(+e.target.value)};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')document.querySelector('#prev').click();if(e.key==='ArrowRight')document.querySelector('#next').click();if(e.key===' '){e.preventDefault();playing?pause():play(false)}});init().catch(error=>document.body.innerHTML=`<pre>${error.stack}</pre>`);
</script></body></html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        manifest: dict[str, Any],
        videos: dict[str, Path],
        **kwargs: Any,
    ) -> None:
        self.manifest = manifest
        self.videos = videos
        super().__init__(*args, directory=str(Path.cwd()), **kwargs)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/manifest":
            self._send_bytes(
                json.dumps(self.manifest, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
            return
        if path.startswith("/video/"):
            video = self.videos.get(path.removeprefix("/video/"))
            if video is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(video, include_body=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        path = unquote(urlparse(self.path).path)
        video = self.videos.get(path.removeprefix("/video/")) if path.startswith("/video/") else None
        if video is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(video, include_body=False)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path, *, include_body: bool) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            unit, requested = range_header.split("=", 1)
            if unit != "bytes":
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, _, last = requested.partition("-")
            start = int(first) if first else max(0, size - int(last))
            end = min(int(last) if last else size - 1, size - 1)
            if start < 0 or start >= size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not include_body:
            return
        with path.open("rb") as file:
            file.seek(start)
            self.wfile.write(file.read(length))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[l4-trace-viewer] {self.address_string()} {format % args}")


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def server_class_for_host(host: str) -> type[ThreadingHTTPServer]:
    return IPv6ThreadingHTTPServer if ":" in host else ThreadingHTTPServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    trace_dir = args.trace_dir.resolve()
    video_dir = args.video_dir.resolve()
    videos = _find_videos(video_dir, args.episode_index)
    manifest = build_manifest(
        trace_dir,
        video_dir,
        episode_index=args.episode_index,
    )
    handler = partial(ViewerHandler, manifest=manifest, videos=videos)
    server = server_class_for_host(args.host)((args.host, args.port), handler)
    print(f"[l4-trace-viewer] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
