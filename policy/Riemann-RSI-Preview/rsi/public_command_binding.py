"""Evaluator binding; the actor receives detached public frames and numerical planning only."""
from rsi.rule_configuration import official_root as _rule_official_root
from copy import deepcopy
import importlib
import json
from pathlib import Path
from harness.runtime.public_policy_recording import STATE_WIDTHS, CAMERAS
from skills.shared.orchestration.public_command_episode.implementation import PublicCommandActor, PublicCommandEvidence as PublicEvidence


MOTION_SPEED = 1.8


def wire_observation(raw,instruction):
    return {'instruction':instruction,'state':{key:deepcopy(raw['state'][key]) for key in STATE_WIDTHS},
        'vision':{name:{'color':view['colors'].copy()} for name in CAMERAS for view in [raw['vision'][name]]}}


def diagnostics(actor):return actor.result()


class PublicCommandBinding:
    """Task-local context for the identical public command protocol binding."""

    def __init__(self, task_root):
        self.task_root = Path(task_root).resolve()

    def verify_resources(self,task):
        if Path(task).resolve()!=self.task_root:raise ValueError('Use the canonical task in this checkout')
        from rsi.task_contract import verify
        verify(self.task_root.parents[1],self.task_root.name)
        declaration=json.loads((self.task_root/'bindings/source-binding.json').read_text())
        from rsi.rule_runtime import verify_official_files
        verify_official_files(_rule_official_root(),self.task_root.name)
        return declaration

    def run(self,env,recorder,declaration,exchange,args,actors):
        if env.step_lim!=declaration['official_horizon'] or env.dt!=.004 or env.obs_manager.collect_interval!=10:
            raise ValueError('Official task horizon or public clock changed')
        planner=importlib.import_module(declaration['motion_module']).MaturePlanner(self.task_root.parents[1])
        instruction=recorder.public.instruction
        if not isinstance(instruction,str) or not instruction:raise ValueError("Missing actual same-frame official public instruction")
        actor=PublicCommandActor(self.task_root.name,wire_observation(recorder.public.last_observation,instruction),declaration,planner)
        actors.append(actor);exchange.phase=lambda:actor.phase
        try:
            while not env.end_flag[0]:
                exchange(actor.next_action(wire_observation(recorder.public.last_observation,instruction)))
        finally:
            actor.close()
