"""The return reserve (OURS_RETURN_RESERVE, official conditions): schedule by the remaining steps, keep the return.

ReturnReserveMixin goes in front of the official worker class (agp_core/official_worker.py worker_class, after
agp_core/speed_profiles.py and agp_core/handover.py), over agp_core/approx_home.py's ApproxHomeMixin, which official conditions
always compose (OURS_RGB_ONLY). With the switch off it is not composed and nothing here runs.

The estimated return (return_steps): the control steps a home of both arms would take now, planned exactly as the
approximate home of a step that homes both arms plans it (ApproxHomeMixin, with F's time law when the gripper is empty,
agp_core/speed_profiles.py), at its first tier (STANDARD margins, joints only toward their start angles; the remaining
budget is not handed to that planning, so a short budget never lowers the estimate to the LAST_CHANCE tier and the
planning never writes a budget refusal); the larger of the two arms (they home in parallel); 0 for an arm already back
(inside the return tolerance with the smallest END margin, ApproxHomeMixin._other_is_back); None when a home cannot be
planned. An arm's plan is made again only when its joints (rounded to 1e-4 rad, CACHE_DECIMALS), its gripper command
or the official reading of its fingers (rounded to 0.01) changed since it was made, and a plan whose joints may leave
their start intervals (the free attempts, which look at the other arm) also when the other arm's changed; so holds,
observations without motion and the arm a step leaves alone reuse it (planning one arm takes about 0.3 to 1 s).

Every observation (the worker's observe, i.e. every observation the runtime returns after a step or an observe) gets
state.return_reserve (STATE_KEY): return_steps, margin_steps (RESERVE_MARGIN_STEPS), spare_steps = steps_left -
return_steps - margin_steps, releases (goal releases completed so far), steps_per_release = steps used so far /
releases, more_releases = floor(spare_steps / steps_per_release) (0 when spare_steps <= 0; None without a release or an
estimate). agp_core/compact.py shows it as `budget`.

A goal release (counted): a gripper open that executed (ok, motion executed, not unreached) of an arm whose dispatched
args carry RELEASE_KEY. The runtime (agp_core/runtime_ext.py DeclareMixin) adds that key to an open whose evidence is a
release at the goal: release_purpose goal (the default) and a release_requirement other than "empty" (an open of an
empty gripper, or a recovery set-down, is not counted). Grip width (OURS_GRIP_WIDTH): a width action that is a
release (GripWidthMixin) carries the key under the same rule and counts like an open once it reached its gap.

The reserve (before any motion, after the stage is prepared, agp_core/speed_profiles.py's F-g delay included): a stage in
which some arm does anything other than `home` (a move, a gripper action, a hold) is refused with REFUSAL when its
scheduled samples are more than spare_steps, i.e. when the steps left after it would no longer cover the estimated
return plus the margin, unless every such arm's dispatched args carry FINAL_KEY: the runtime adds it when the arm's
evidence has `final` (the last action before the return) or `no_return` (the task needs no return). A stage of homes
only is never refused here (the approximate home has its own budget rule). An unknown estimate refuses nothing. The
refusal is the worker's usual before-motion answer (ok false, motion_started false, no sample), so the runtime and the
run treat it like any other refusal.

Records: return_reserve.json (the constants), return_reserve.jsonl (every estimate with its planning time, every
refusal, every declared stage that needed more than the spare steps, every counted release).
"""
from __future__ import annotations

import json
import time

import numpy as np

STATE_KEY = "return_reserve"
RESERVE_MARGIN_STEPS = 10       # engineering value: a home's settle samples (up to 3) and the return's growth over a
                                # short move; stated in the prompt and in every refusal
FINAL_KEY = "ours_final"        # set by the runtime in an arm's dispatched args: "final" or "no_return"
RELEASE_KEY = "ours_release"    # set by the runtime on a goal release (an open)
FINAL_EVIDENCE = ("final", "no_return")
CACHE_DECIMALS = 4
REFUSAL = ("Return reserve: this step needs {need} control steps, but only {spare} can be spent before the return: "
           "{left} are left, a home of both arms needs {ret} and the margin is {margin}. Home both arms now, or give "
           "the step evidence final (the last action before the return) or no_return (the task needs no return).")


def reserve_state(used, left, estimate, releases, margin=RESERVE_MARGIN_STEPS):
    """The observation entry (module docstring) from the steps used and left, the estimate and the releases."""
    ret = None if estimate is None else estimate.get("return_steps")
    spare = None if ret is None or left is None else int(left) - int(ret) - int(margin)
    per = round(used / releases, 1) if releases and used is not None else None
    more = None
    if per and spare is not None:
        more = max(0, int(spare // (used / releases))) if spare > 0 else 0
    return {"return_steps": ret, "margin_steps": int(margin), "spare_steps": spare, "releases": int(releases),
            "steps_per_release": per, "more_releases": more}


def reserve_verdict(plans, left, ret, margin=RESERVE_MARGIN_STEPS):
    """(refuse, record) for prepared plans {arm: {command, args, run_samples}} with `left` steps and the estimate."""
    homes_only = all(p["command"] == "home" for p in plans.values())
    others = sorted(arm for arm, p in plans.items() if p["command"] != "home")
    declared = {arm: (plans[arm].get("args") or {}).get(FINAL_KEY) for arm in others}
    need = max(int(p["run_samples"]) for p in plans.values())
    spare = None if ret is None or left is None else int(left) - int(ret) - int(margin)
    record = {"commands": {arm: p["command"] for arm, p in plans.items()}, "need": need, "left": left,
              "return_steps": ret, "margin_steps": int(margin), "spare_steps": spare,
              "declared": {arm: v for arm, v in declared.items() if v}}
    if homes_only or spare is None or need <= spare:
        return False, dict(record, verdict="return" if homes_only else "within spare" if spare is not None
                           else "no estimate")
    if others and all(declared.values()):
        return False, dict(record, verdict="declared")
    return True, dict(record, verdict="refused",
                      error=REFUSAL.format(need=need, spare=max(0, spare), left=left, ret=ret, margin=int(margin)))


class ReturnReserveMixin:
    """In front of the official worker class (module docstring)."""

    def __init__(self, env, session):
        super().__init__(env, session)
        from migration import worker as base
        self._rr_cache = None
        self._rr_releases = 0
        self._rr_estimating = False
        base.atomic(self.session / "return_reserve.json", {
            "switch": "OURS_RETURN_RESERVE", "state_key": STATE_KEY, "margin_steps": RESERVE_MARGIN_STEPS,
            "estimate": "a home of both arms planned now by the approximate home (first tier, no budget), the larger "
                        "arm; 0 for an arm already back",
            "refused": "a stage with any non-home command whose scheduled samples exceed spare_steps, unless every "
                       f"such arm carries {FINAL_KEY} (evidence final or no_return)",
            "release": f"an executed open carrying {RELEASE_KEY} (release_purpose goal, release_requirement not empty)",
            "source": "agp_core/return_reserve.py"})

    def _rr_log(self, entry):
        try:
            with (self.session / "return_reserve.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(entry, count=int(self.env.take_action_cnt[0])), default=float) + "\n")
        except Exception:                                    # noqa: BLE001 - a failed record never stops the worker
            pass

    # -- the estimate
    def _remaining_budget(self):
        if self._rr_estimating:                              # the estimate plans the first tier, without a budget
            return None
        return super()._remaining_budget()

    def _rr_key(self, arm):
        """What one arm's estimate depends on: its joints (rounded), its gripper command and the official reading of
        its fingers (F's time law follows whether the gripper is empty)."""
        context = self.arms[arm]
        reading = None
        observed = getattr(self, "_sp_observed", None)
        if callable(observed):
            value = observed(arm)
            reading = None if value is None else round(float(value), 2)
        return json.dumps({"joints": np.round(np.asarray(self.rm.get_joint(context["robot"])[0], float),
                                              CACHE_DECIMALS).tolist(),
                           "grip": float(self.grips[context["robot"].arm_name]), "reading": reading})

    def _rr_estimate(self):
        """The estimated return (module docstring). Per arm the plan is reused while its key (_rr_key) is unchanged;
        a plan whose joints may leave their start intervals (joint_range free, which also looks at the other arm) is
        reused only while the other arm's key is unchanged as well."""
        keys = {arm: self._rr_key(arm) for arm in self.arms}
        cache = self._rr_cache if isinstance(self._rr_cache, dict) else {}
        started = time.monotonic()
        arms, planned = {}, []
        saved = self._approx_arms
        self._approx_arms, self._rr_estimating = set(self.arms), True
        try:
            for arm in self.arms:
                others = {o: keys[o] for o in self.arms if o != arm}
                hit = cache.get(arm)
                if hit is not None and hit["key"] == keys[arm] and (hit["others"] is None or hit["others"] == others):
                    arms[arm] = hit["entry"]
                    continue
                planned.append(arm)
                if self._other_is_back(arm, None):
                    entry = {"back": True, "steps": 0}
                else:
                    try:
                        plan = self._prepare(arm, "home", {})
                    except Exception as exc:                 # noqa: BLE001 - no plan: the estimate is unknown
                        plan = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    if plan.get("ok"):
                        info = plan.get("approx_home") or {}
                        entry = {"back": False, "steps": len(plan["points"]), "tier": info.get("tier"),
                                 "method": info.get("method"), "joint_range": info.get("joint_range")}
                    else:
                        entry = {"back": False, "steps": None, "error": str(plan.get("error"))[:300]}
                arms[arm] = entry
                cache[arm] = {"key": keys[arm], "entry": entry,
                              "others": others if entry.get("joint_range") == "free" else None}
        finally:
            self._approx_arms, self._rr_estimating = saved, False
        self._rr_cache = cache
        steps = [a["steps"] for a in arms.values()]
        estimate = {"return_steps": None if any(s is None for s in steps) else max(steps), "arms": arms}
        if planned:
            self._rr_log({"event": "estimate", **estimate, "planned": planned,
                          "seconds": round(time.monotonic() - started, 3)})
        return estimate

    # -- the observation
    def observe(self):
        out = super().observe()
        state = out.get("state") if isinstance(out, dict) and out.get("ok") else None
        if isinstance(state, dict):
            try:
                estimate = self._rr_estimate()
            except Exception as exc:                         # noqa: BLE001 - never fails an observation
                estimate = {"return_steps": None, "error": f"{type(exc).__name__}: {exc}"}
            state[STATE_KEY] = reserve_state(state.get("low_level_actions"), state.get("remaining_actions"),
                                             estimate, self._rr_releases)
        return out

    # -- the reserve
    def _prepare_stage(self, actions, starts=None):
        prepared = super()._prepare_stage(actions, starts)
        if starts is not None or not prepared.get("ok"):
            return prepared
        plans = prepared["plans"]
        if all(p["command"] == "home" for p in plans.values()):
            return prepared
        estimate = self._rr_estimate()
        refuse, record = reserve_verdict(plans, self._remaining_budget(), estimate.get("return_steps"))
        if refuse or record["verdict"] == "declared":
            self._rr_log({"event": "stage", "sequence": self.motion_sequence, **record})
        if refuse:
            return {"ok": False, "error": record["error"], "ours_return_reserve": record}
        return prepared

    # -- the releases
    def execute_actions(self, actions):
        out = super().execute_actions(actions)
        try:
            for arm, action in (actions or {}).items():
                args = (action or {}).get("args") or {}
                outcome = ((out or {}).get("arms") or {}).get(arm) or {}
                if (action.get("command") == "gripper" and args.get("action") in ("open", "width")   # (width: the
                        and args.get(RELEASE_KEY)                                                   # grip width)
                        and outcome.get("ok") and outcome.get("motion_executed")
                        and outcome.get("target_reached") is not False):
                    self._rr_releases += 1
                    self._rr_log({"event": "release", "arm": arm, "releases": self._rr_releases,
                                  "sequence": self.motion_sequence})
        except Exception:                                    # noqa: BLE001 - the count never fails a stage
            pass
        return out


def worker_class(base_cls):
    """`base_cls` with ReturnReserveMixin in front (official_worker composes it only with OURS_RETURN_RESERVE on; the
    class needs ApproxHomeMixin, i.e. OURS_RGB_ONLY)."""
    from agp_core import approx_home
    if not issubclass(base_cls, approx_home.ApproxHomeMixin):
        raise ValueError("OURS_RETURN_RESERVE needs the approximate home (OURS_RGB_ONLY)")
    return type(f"ReturnReserve{base_cls.__name__}", (ReturnReserveMixin, base_cls),
                {"__module__": __name__, "__doc__": f"{base_cls.__qualname__} with the return reserve (G4)."})
