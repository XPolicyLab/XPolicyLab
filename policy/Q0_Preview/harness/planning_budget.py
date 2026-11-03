"""Planning response audit only; no call quotas or recovery allowances."""


class PlanningResponseLog:
    def __init__(self):
        self.responses = 0

    def record(self, outcomes, status='tools'):
        self.responses += 1
        return {'response_index': self.responses, 'reason': status,
                'outcomes': outcomes}
