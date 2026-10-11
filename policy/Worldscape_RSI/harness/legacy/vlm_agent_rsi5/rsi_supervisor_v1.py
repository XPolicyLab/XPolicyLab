"""Pure collection supervisor. No simulator, reward mutation, vision or I/O.

Task adapters supply timestamped evidence from their existing observation/reward
path. This module never evaluates hidden task predicates or invents observations.
"""
from dataclasses import dataclass, field, asdict
from math import isfinite
from typing import Optional

VERSION = '1.0.0'
LIMITS = {'general_pickup': 200, 'solve_equation': 300,
          'stack_blocks_by_language': 400, 'align_blocks': 200,
          'classify_objects_by_language': 1100, 'pour_by_language': 800,
          'store_tools_in_toolbox': 900, 'pick_from_conveyor_by_image': 700}


@dataclass(frozen=True)
class Evidence:
    value: Optional[bool] = None
    step: Optional[int] = None
    source: str = ''

    def at(self, step):
        return self.value if self.step == step and self.source else None


@dataclass(frozen=True)
class Snapshot:
    step: int
    score: Optional[float] = None
    score_step: Optional[int] = None
    score_source: str = ''
    score_scale: str = 'unknown'
    terminal: bool = False
    native_success: Optional[bool] = None
    # Identifier of a REAL native score-evaluation event, not merely a read.
    score_event: Optional[str] = None
    trigger: Evidence = field(default_factory=Evidence)
    current: dict = field(default_factory=dict)
    historical: dict = field(default_factory=dict)

    def fresh_score(self):
        if self.score_step != self.step or not self.score_source:
            return None
        if self.score is None or not isfinite(self.score):
            return None
        return self.score


@dataclass(frozen=True)
class StageContract:
    name: str
    # Only known, task-audited scoring stages may set these thresholds.
    expected_score: Optional[float] = None
    expected_delta: Optional[float] = None
    score_scale: str = 'unknown'
    required_current: tuple = ()
    required_historical: tuple = ()
    require_trigger: bool = True
    settle_steps: int = 0
    max_recoveries: int = 1
    restart_on_scored_miss: bool = True


class Supervisor:
    def __init__(self, task, native_limit):
        if task not in LIMITS or native_limit != LIMITS[task]:
            raise ValueError('Native limit does not match audited task contract')
        self.task, self.limit = task, native_limit
        self.previous = None
        self.last_step = -1
        self.stage = None
        self.stage_baseline = None
        self.stage_start = None
        self.stage_score_event = None
        self.recoveries = 0
        self.costs = {}

    def feedback(self, snapshot):
        if snapshot.step < self.last_step or snapshot.step < 0:
            raise ValueError('Out-of-order or invalid native step')
        self.last_step = snapshot.step
        score = snapshot.fresh_score()
        delta = None if score is None or self.previous is None else score-self.previous
        if score is not None:
            self.previous = score
        return dict(schema='rsi.supervisor.v1', version=VERSION,
                    native_step=snapshot.step, native_step_limit=self.limit,
                    remaining_steps=max(0, self.limit-snapshot.step),
                    current_score=score, delta_score=delta,
                    score_scale=snapshot.score_scale,
                    score_status='fresh' if score is not None else 'unknown',
                    score_source=snapshot.score_source,
                    score_event=snapshot.score_event,
                    trigger=asdict(snapshot.trigger),
                    current={k: asdict(v) for k,v in snapshot.current.items()},
                    historical={k: asdict(v) for k,v in snapshot.historical.items()},
                    native_success=(snapshot.native_success if snapshot.terminal else None),
                    terminal=snapshot.terminal)

    def begin_stage(self, contract, snapshot):
        if contract.settle_steps < 0 or contract.max_recoveries < 0:
            raise ValueError('Negative stage bound')
        if contract.expected_score is not None or contract.expected_delta is not None:
            if contract.score_scale == 'unknown':
                raise ValueError('Scored stages require an explicit raw scale')
        self.stage = contract
        self.stage_start = snapshot.step
        self.stage_baseline = snapshot.fresh_score()
        self.stage_score_event = snapshot.score_event
        self.recoveries = 0

    def record_cost(self, kind, before, after):
        if before < 0 or after < before:
            raise ValueError('Invalid executed step interval')
        self.costs.setdefault(kind, []).append(after-before)

    def budget(self, snapshot, *, planned_steps=None, finish_reserve=0,
               is_finishing=False, includes_settle=False, kind=None):
        """planned_steps is the adapter's full primitive estimate/bound.

        includes_settle must cover closure/hold as well as planned motion.
        Measured maxima are ADVISORY, never a guarantee of future cost.
        Native per-control-step truncation must remain installed in executor.
        """
        remaining = max(0, self.limit-snapshot.step)
        if finish_reserve < 0 or (planned_steps is not None and planned_steps < 0):
            raise ValueError('Negative budget')
        if snapshot.terminal:
            return {'decision':'stop', 'reason':'native_terminal', 'remaining':remaining}
        if not remaining:
            return {'decision':'stop', 'reason':'native_budget_exhausted', 'remaining':0}
        advisory = max(self.costs.get(kind, []), default=None)
        reserve = 0 if is_finishing else finish_reserve
        ceiling = max(0, remaining-reserve)
        if planned_steps is None or not includes_settle:
            return {'decision':'needs_estimate', 'reason':'full_primitive_cost_unknown',
                    'remaining':remaining, 'step_ceiling':ceiling,
                    'observed_max':advisory}
        return {'decision':'allow' if planned_steps <= ceiling else 'replan',
                'reason':'fits_budget' if planned_steps <= ceiling else 'insufficient_budget_with_finish_reserve',
                'remaining':remaining, 'step_ceiling':ceiling,
                'observed_max':advisory}

    def assess(self, snapshot, *, action_ok, checkpoint_step, phase_complete,
               recovery_fits=False):
        """Called AFTER an action, including zero-step rejection and sync.

        checkpoint_step = end of physical stage action; adapter persists this
        across settling observations, so the window cannot reset indefinitely.
        recover means request a bounded repair; no robot action is generated here.
        """
        c = self.stage
        if c is None:
            raise ValueError('begin_stage required')
        if snapshot.step < checkpoint_step or checkpoint_step < self.stage_start:
            raise ValueError('Invalid stage checkpoint')
        if snapshot.terminal:
            return self._result('stop_success' if snapshot.native_success is True and snapshot.step <= self.limit else 'stop_failure',
                                'native_terminal')
        if snapshot.step >= self.limit:
            return self._result('stop_failure', 'native_budget_exhausted')
        # Native termination takes precedence over interrupted/truncated motion.
        if action_ok is not True:
            return self._repair('execution_failed_or_unknown', recovery_fits)
        if not phase_complete:
            return self._result('continue_stage', 'phase_actions_incomplete')
        if snapshot.step-checkpoint_step < c.settle_steps:
            return self._result('needs_observation', 'bounded_settling_window',
                                settle_steps_remaining=c.settle_steps-(snapshot.step-checkpoint_step))
        checks = [snapshot.current.get(k, Evidence()).at(snapshot.step) for k in c.required_current]
        for k in c.required_historical:
            e = snapshot.historical.get(k, Evidence())
            checks.append(e.value if e.source and e.step is not None and e.step <= snapshot.step else None)
        # Unknown is NOT a physical failure or a reason for an automatic restart.
        if any(v is False for v in checks):
            return self._repair('postcondition_failed', recovery_fits)
        if any(v is None for v in checks):
            return self._result('needs_observation', 'postcondition_unknown')
        scored = c.expected_score is not None or c.expected_delta is not None
        if scored:
            score = snapshot.fresh_score()
            if score is None or snapshot.score_scale != c.score_scale:
                return self._result('needs_observation', 'score_unknown_or_scale_mismatch')
            if c.require_trigger and snapshot.trigger.at(snapshot.step) is not True:
                return self._result('needs_observation', 'scoring_trigger_not_confirmed')
            if snapshot.score_event is None or snapshot.score_event == self.stage_score_event:
                return self._result('needs_observation', 'no_new_native_score_event')
            if c.expected_delta is not None and self.stage_baseline is None:
                return self._result('needs_observation', 'stage_score_baseline_unknown')
            passed = (c.expected_score is None or score >= c.expected_score)
            passed &= c.expected_delta is None or score-self.stage_baseline >= c.expected_delta
            if not passed:
                if c.restart_on_scored_miss:
                    return self._result('stop_failure', 'scored_stage_missed')
                return self._repair('scored_stage_missed', recovery_fits)
        elif not checks:
            return self._result('needs_observation', 'no_verified_stage_criterion')
        return self._result('advance_stage', 'stage_passed')

    def _repair(self, reason, fits):
        if fits and self.recoveries < self.stage.max_recoveries:
            self.recoveries += 1
            return self._result('recover', reason, recovery_index=self.recoveries)
        return self._result('stop_failure', reason)

    def _result(self, decision, reason, **extra):
        return dict(decision=decision, reason=reason, stage=self.stage.name, **extra)
