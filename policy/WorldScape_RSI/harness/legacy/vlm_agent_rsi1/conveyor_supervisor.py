"""Private conveyor adapter. No reward calls, object state, or quality filters."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

RELEASE_SHA256 = '1c016b898de743e884ce8fe26aa28365c3a1db1c01998282b2f619411f094bb0'
_path = Path(__file__).with_name('rsi_supervisor_v1.py')
if hashlib.sha256(_path.read_bytes()).hexdigest() != RELEASE_SHA256:
    raise RuntimeError('Frozen supervisor release hash mismatch')
_name = '_conveyor_frozen_rsi_supervisor_v1'
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
core = sys.modules[_name]

STAGES = ('basket', 'intercept', 'transfer', 'place', 'lift', 'finish')
CHECKS = {'basket': ('basket_retained', 'basket_prepositioned'),
          'intercept': ('target_retained',), 'transfer': ('over_basket',),
          'place': ('target_inside_left_clear',)}
FINISH_RESERVE = 60  # Planning reserve, NOT a guarantee; actual final plan is checked.

def enabled():
    return (os.environ.get('ROBODAWN_CONVEYOR_SUPERVISOR') == '1'
            and os.environ.get('ROBODAWN_CONVEYOR_REWARD_FEEDBACK') == '1'
            and os.environ.get('ROBODAWN_RUN_MODE') == 'collection')

def prompt_text(packet):
    if not packet:
        return ''
    return ('\nCOLLECTION SUPERVISOR (sidecar only):\n' + json.dumps(packet) +
        '\nUnknown score/trigger is not failure and must not cause repeated waits. '
        'Before changing stages, provide supervisor_evidence: a mapping from each required '
        'check to {value: true/false/null, step: CURRENT native step, source: "rgb", '
        'reason: concrete current-image evidence}. Gripper opening alone is not retention. '
        'Required checks: basket -> basket_retained,basket_prepositioned; intercept -> '
        'target_retained; transfer -> over_basket; place -> target_inside_left_clear. '
        'Intermediate commands keep same stage; program checks full planned cost plus holds. '
        'Budget replan requires a shorter valid operation, never increasing the native700 cap.')

class ConveyorSupervisor:
    def __init__(self, root, native_limit):
        self.supervisor = core.Supervisor('pick_from_conveyor_by_image', native_limit)
        self.path = Path(root)/'conveyor_supervisor.jsonl'
        if self.path.exists():
            raise FileExistsError(self.path)
        self.snapshot = None
        self.stage = None
        self.checkpoint = 0
        self.latest = None

    def _write(self, event):
        with self.path.open('a') as f:
            f.write(json.dumps(event, allow_nan=False)+'\n')
        self.latest = event
        return event

    def read(self, packet, evidence=None):
        step = packet['native_step']
        supported = packet.get('status') == 'ok'
        terminal = packet.get('terminal') is True
        success = terminal and supported and packet.get('native_success') is True
        # Existing sidecar sequence/source_step means a READ, not a native evaluation.
        # Only a supported native success transition proves an actual final scoring event.
        initial = supported and packet.get('action_id') == 'baseline' and step == 0
        score = 1.0 if success else (0.0 if initial else None)
        current = {}
        for key, value in (evidence or {}).items():
            if (key in {c for checks in CHECKS.values() for c in checks}
                    and isinstance(value, dict) and value.get('source') == 'rgb'
                    and value.get('reason') and type(value.get('value')) in (bool, type(None))):
                current[key] = core.Evidence(value.get('value'), value.get('step'),
                    'model RGB assessment: '+str(value['reason']))
        hist = packet.get('phase_conditions', {}).get('native_and_group', {}).get('historical')
        self.snapshot = core.Snapshot(step=step, score=score,
            score_step=step if score is not None else None,
            score_source='audited native terminal success' if success else ('initial binary baseline' if initial else ''),
            score_scale='0..1', terminal=terminal,
            native_success=packet.get('native_success') if supported and terminal else None,
            score_event=f'native_success:{step}' if success else None,
            trigger=core.Evidence(True, step, 'native terminal success transition') if success else core.Evidence(),
            current=current, historical={'native_and_group':core.Evidence(hist, step, 'native check_list history' if supported else '')})
        self.packet = packet
        return self.snapshot

    def _begin(self, stage):
        self.stage = stage
        contract = core.StageContract(stage, expected_score=1.0 if stage in ('lift','finish') else None,
            score_scale='0..1', required_current=CHECKS.get(stage, ()),
            require_trigger=stage in ('lift','finish'), max_recoveries=1)
        self.supervisor.begin_stage(contract, self.snapshot)
        self.checkpoint = self.snapshot.step

    def boundary(self, packet, result=None):
        snap = self.read(packet)
        if self.stage is None:
            self._begin('basket')
        assessment = None
        if result is not None:
            before = result.get('steps_before', snap.step)
            self.supervisor.record_cost(result.get('kind','unknown'), before, snap.step)
            self.checkpoint = snap.step
            if result.get('supervisor_budget',{}).get('decision') in ('replan','needs_estimate'):
                assessment = {'decision':'replan','reason':'full_primitive_budget_not_approved','stage':self.stage}
            else:
                assessment = self.supervisor.assess(snap, action_ok=result.get('ok'),
                    checkpoint_step=self.checkpoint, phase_complete=False,
                    recovery_fits=700-snap.step >= FINISH_RESERVE+10)
        return self._write({'kind':'primitive_boundary' if result else 'snapshot',
            'release_sha256':RELEASE_SHA256,'feedback':self.supervisor.feedback(snap),
            'action_acceptance':assessment,'command':result.get('command') if result else None,
            'budget':result.get('supervisor_budget') if result else None,
            'actual_cost':None if result is None else snap.step-result.get('steps_before',snap.step)})

    def begin_turn(self, context):
        context = context or {}
        last = (self.latest or {}).get('action_acceptance') or {}
        if last.get('decision') in ('stop_failure', 'stop_success'):
            return last
        requested = context.get('stage') or self.stage
        self.read(self.packet, context.get('supervisor_evidence'))
        if requested in ('recover', self.stage):
            return None
        if requested not in STAGES or STAGES.index(requested) != STAGES.index(self.stage)+1:
            gate = {'decision':'needs_observation','reason':'invalid_stage_transition','stage':self.stage}
        else:
            gate = self.supervisor.assess(self.snapshot, action_ok=True,
                checkpoint_step=self.checkpoint, phase_complete=True,
                recovery_fits=700-self.snapshot.step >= FINISH_RESERVE+10)
            if gate['decision'] == 'advance_stage':
                self._begin(requested)
        self._write({'kind':'stage_gate','feedback':self.supervisor.feedback(self.snapshot),
                     'action_acceptance':gate,'requested_stage':requested})
        return None if gate['decision'] == 'advance_stage' else gate

    def budget(self, planned_steps, kind, is_finishing=False):
        b = self.supervisor.budget(self.snapshot, planned_steps=planned_steps,
            finish_reserve=FINISH_RESERVE, includes_settle=True,
            is_finishing=is_finishing, kind=kind)
        self._write({'kind':'budget_preflight','native_step':self.snapshot.step,
                     'planned_steps_including_settle':planned_steps,'budget':b})
        return b
