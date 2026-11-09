"""Reuse the published simulation motion port, with the actual task budget.

No new distance, arrival-error, retry-count or per-segment time gates. The
experimental contact policy is exactly fill_pen_holder c008's policy: native
physics/planner and joint/self/cross-arm checks remain authoritative. It is not
an object-aware collision planner or hardware-safety qualification.
"""
from orimapi.skills.code_skills.fill_pen_holder.motion_adapter import adapter_class


def api_class():
    base = adapter_class()  # Private provider namespace; no shared mutation.

    class EpisodeAPI(base):
        @property
        def over(self):
            return self.owner.cancelled or self.owner.latest.step >= self.owner.budget

        def sim_time_left(self):
            return max(0, self.owner.budget - self.owner.latest.step) / 25.0

        def estimate_tcp_chain(self, arm, stages):
            result = super().estimate_tcp_chain(arm, stages)
            if "estimated_total_steps" in result:
                remaining = max(0, self.owner.budget - self.owner.latest.step)
                fits = result["estimated_total_steps"] <= remaining
                result.update(
                    estimate_ok=fits, remaining_action_steps=remaining,
                    failed_stage=None if fits else "budget",
                    reason=None if fits else "insufficient_action_budget",
                )
            return result

        def move_tcp(self, arm, target, feedback):
            # The inherited provider's 3mm/2deg arrival veto is NOT reinstated.
            return self.move_stage(arm, "contact", target, feedback)

    return EpisodeAPI
