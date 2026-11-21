"""Start one continuing controller for the whole evaluation stream, no workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from client import atomic_json, chain_tip


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--codex', required=True, type=Path)
    p.add_argument('--skill-root', required=True, type=Path)
    p.add_argument('--stream-root', required=True, type=Path)
    p.add_argument('--experience-root', required=True, type=Path)
    p.add_argument('--model', default='gpt-6.1-sol')
    p.add_argument('--effort', default='xhigh')
    a = p.parse_args()
    skill, stream, experience = [x.resolve() for x in (a.skill_root, a.stream_root, a.experience_root)]
    tip = chain_tip(skill)
    stream.mkdir(parents=True, exist_ok=True)
    lock = stream / 'controller-active.json'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, json.dumps({'supervisor_pid': os.getpid(), 'memory_before': tip}).encode())
    os.close(fd)
    here = Path(__file__).resolve().parent
    agent = stream / 'agent'
    agent.mkdir(exist_ok=True)
    io = here / 'agent_io.py'
    prompt = f'''You are the sole continuing Cypher Tempre robot controller for one sequential RoboDojo evaluation stream. This is one agent across episodes, with no workers/subagents/search workers/other model calls. Model={a.model}, effort={a.effort}. Your persistent skill and complete existing chain/registries/generated tools are at {skill}. Your existing benchmark experience and helpers are at {experience}. These remain available across the whole run. Read {skill / 'SKILL.md'}, verify this existing chain and read genesis; continue it. Never substitute a new empty chain. Keep Cambium auto-sprout on. The user explicitly authorizes generating, proposing, activating, integrating and executing needed faculties in-session; no per-source human review is claimed.

The observation-only transport is: {sys.executable} {io} --stream-root {stream} observe --wait 20
It returns the current session and turn, embodiment instructions, goal, own RGB paths, named EEF/joints, recent arrival feedback and allowed tools. View the actual RGB. The only native tools are move_eef and give_up. Estimate object positions from RGB. Use normal coding and all your existing/generated faculties; compute useful measurements from observations BEFORE choosing the action they should inform. Compare model-scored perspectives when a new difficulty warrants it. Record experience and corrected beliefs in your continuing Timechain, using actual prior rings, rather than repeating initialization. Never claim a commanded arrival or closed gripper proves object retention.

Write exactly one model-authored {{"name":...,"arguments":...}} JSON to your own reply file, then transmit with: {sys.executable} {io} --stream-root {stream} act SESSION TURN REPLY_FILE
Check the next observation before assuming motion succeeded. The transport validates syntax only and makes no robot decisions. A waiting response is not terminal: wait on this same stream. episodes lists ended-session receipts; they are boundary receipts, not success scores. A new session is a new robot scene, while your Timechain, faculties, generated code, and learning continue. A stream_stopped response ends the controller. Retain failures and uncertainty.

Allowed: current stream, your existing skill/experience, public static robot/task/reward descriptions and code. Prohibited: hidden live object/root poses, depth, segmentation, contacts, native planner-path queries, hidden verification layouts, or modifying benchmark physics/reward/planner/action limits. Do not read credentials, unrelated private user files, or other applications. Do not switch between preselected task-specific policy checkpoints; this same model and inference procedure apply to every task. No new simulator runs, native restarts, external solutions, or additional model workers. Honor the supplied native budgets. All newly learned experience remains in this same agent's persistent store.'''
    executable = a.codex.resolve()
    version = subprocess.run([str(executable), '--version'], capture_output=True, text=True, check=True).stdout.strip()
    atomic_json(stream / 'controller-start.json', {'model': a.model, 'effort': a.effort,
                'skill_root': str(skill), 'experience_root': str(experience), 'memory_before': tip,
                'cli_version': version, 'cli_sha256': hashlib.sha256(executable.read_bytes()).hexdigest()})
    (stream / 'controller-prompt.txt').write_text(prompt, encoding='utf-8')
    command = [str(a.codex.resolve()), 'exec', '--ignore-user-config', '--skip-git-repo-check', '--cd', str(agent),
               '--model', a.model, '--config', f'model_reasoning_effort="{a.effort}"',
               '--config', 'project_doc_max_bytes=0', '--config', 'skills.bundled.enabled=false',
               '--config', 'skills.include_instructions=false', '--config', 'features.skip_host_skill_discovery=true',
               '--config', 'features.hooks=false', '--config', 'features.multi_agent=false', '--config', 'features.apps=false',
               '--config', 'approval_policy="never"', '--sandbox', 'danger-full-access', '--json', '-']
    child = None
    try:
        with (stream / 'controller-events.jsonl').open('x', encoding='utf-8') as out, (stream / 'controller-stderr.log').open('x', encoding='utf-8') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, encoding='utf-8')
            atomic_json(lock, {'supervisor_pid': os.getpid(), 'controller_pid': child.pid, 'memory_before': tip})
            child.stdin.write(prompt)
            child.stdin.close()
            result = child.wait()
        atomic_json(stream / 'controller-ended.json', {'exit': result, 'memory_after': chain_tip(skill)})
        return result
    finally:
        # Never drop the duplicate-controller guard while its owned process is live.
        if child is None or child.poll() is not None:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
