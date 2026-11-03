"""Bind an existing task's IK requests to its current public observation."""
from physicalrsi_observed_geometry import GeometryUnavailable

class ObservedIKContext:
    def __init__(self,solver,owner):
        self.solver=solver
        self.owner=owner

    def solve(self,arm,seed,target):
        support=getattr(self.owner,'observed_support',None)
        if not support:
            raise GeometryUnavailable('Observe support before language-pour IK')
        return self.solver.solve(arm,seed,target,observed_state=self.owner.obs['state'],
                                 support=support,env_idx=self.owner.env_idx)

    @property
    def last_support_check(self):
        return getattr(self.solver,'last_support_check',None)
