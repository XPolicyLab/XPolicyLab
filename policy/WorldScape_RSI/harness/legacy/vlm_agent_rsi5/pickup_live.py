"""Opt-in private collection supervisor; no reward evaluation calls."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pickup_feedback import PickupFeedback
from pickup_supervisor_adapter import PickupSupervisorAdapter, snapshot
from rsi_supervisor_v1 import Evidence

class LivePickupFeedback(PickupFeedback):
    def __init__(self, env, sidecar):
        super().__init__(env, sidecar)
        self.adapter = PickupSupervisorAdapter()
        self.stage = None
        self.stop_requested = False
        self.turn_gate = None
        self.log_path = Path(sidecar).with_name('supervisor.jsonl')

    def state(self, current=None):
        return snapshot(self.latest, terminal=bool(self.env.end_flag[0]), current=current)

    def log(self, record):
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

    def configure(self, metadata):
        self.turn_gate = None
        metadata = metadata or {}
        requested = metadata.get('stage', self.stage or 'approach')
        if requested not in ('approach', 'grasp', 'lift'):
            requested = self.stage or 'approach'
        if self.stage is None:
            self.stage = 'approach'
            self.adapter.begin(self.stage, self.state())
        if requested != self.stage:
            evidence = {}
            key = 'approach_clear' if self.stage == 'approach' else 'object_retained'
            evidence[key] = Evidence(metadata.get(key), metadata.get('observation_step'),
                                     'current RGB interpreted by collection model')
            packet = self.adapter.after(self.state(evidence), kind='visual_checkpoint', action_ok=True,
                checkpoint_step=self.latest['native_step'], phase_complete=True,
                recovery_fits=False)
            self.log(packet)
            allowed = {'approach':'grasp', 'grasp':'lift'}.get(self.stage)
            if packet['acceptance']['decision'] == 'advance_stage' and requested == allowed:
                self.stage = requested
                self.adapter.begin(self.stage, self.state())
            else:
                self.turn_gate = packet['acceptance']
                self.stop_requested = packet['acceptance']['decision'] == 'stop_failure'
        self.latest['supervisor_stage'] = self.stage

    def preflight(self, actions, extra, kind):
        reserve = 60 if self.stage == 'approach' else 30 if self.stage == 'grasp' else 0
        result = self.adapter.preflight(self.state(), planned_actions=actions,
            max_settle_steps=extra, finish_reserve=reserve,
            is_finishing=self.stage == 'lift', kind=kind)
        if self.turn_gate is not None:
            result = {'decision':'needs_observation', 'reason':'stage_transition_not_verified',
                      'acceptance':self.turn_gate}
        self.log({'native_step':self.latest['native_step'], 'kind':kind,
                  'planned_steps':len(actions)+extra, 'budget':result})
        return result

    def command_done(self, result):
        super().command_done(result)
        if self.stage is None:
            self.configure({})
        if result.get('budget_rejected'):
            self.latest['supervisor'] = result['budget_rejected']
            return
        packet = self.adapter.after(self.state(), kind=result.get('kind','unknown'),
            action_ok=result.get('ok'), checkpoint_step=self.latest['native_step'],
            phase_complete=self.stage == 'lift', recovery_fits=False)
        # A failed action aborts this attempt; no speculative unbounded recovery.
        self.stop_requested = packet['acceptance']['decision'] == 'stop_failure'
        result['supervisor'] = packet
        self.latest['supervisor'] = packet
        self.log(packet)
