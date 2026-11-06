from skills.shared.motion.rigid_transfer_program.implementation import RigidClock, RigidMotion, actual_request, clear_view, own_opening, unavailable
from skills.shared.perception.focus_public_relation_view.implementation import run as focus
from skills.shared.verification.observed_rigid_transfer.implementation import recorded, scene


class RelationViewMotion(RigidMotion):
    def __init__(self, *arguments, **keywords):
        self.view_attempt_recorded = False
        super().__init__(*arguments, **keywords)

    def observe(self, observation):
        super().observe(observation)
        if self.actions and not self.view_attempt_recorded:
            self.current['view_attempts'] = min(4, self.current['view_attempts'] + 1)
            self.view_attempt_recorded = True


def run(api, request, semantic_goal, observe):
    current = recorded(api.command_memory(), api.instruction())
    qualified = semantic_goal(api.instruction())
    if (qualified is None or current['phase'] not in ('clear', 'view') or current['attempts'] >= 3
            or current['job']['empty_observations'] >= 3
            or current['job']['observation_count'] >= 64
            or current['goal'] is not None and current['goal'] != qualified):
        return unavailable(api, request, current)
    current['goal'] = qualified
    if current['phase'] == 'clear':
        return clear_view(api, request, current)
    current['view_hint'] = None
    current['job']['observation_count'] += 1
    observed = observe(api, current)
    if set(observed) != {'scene', 'no_more'}:
        raise ValueError('A public relation producer returns its actual closed scene result')
    if observed['scene'] is None and not observed['no_more']:
        target = focus(api, current)
        if target is not None:
            def moved(program):
                program.current['phase'] = 'view'
                program.current['view_hint'] = None
            return RelationViewMotion(api, actual_request(request, target['side']), current, target['pose'],
                'approach', 1., maximum=96, reviewer=moved)
    def completed(program):
        state = program.current
        state['view_hint'] = None
        result = observe(program.api, state)
        if set(result) != {'scene', 'no_more'}:
            raise ValueError('A settled public relation reuses the same closed observation contract')
        if result['scene'] is not None:
            state['job']['empty_observations'] = 0
            state.update(scene=scene(result['scene']), phase='pregrasp', view_attempts=0)
        elif result['no_more']:
            state.update(phase='home', home_leg=0, view_attempts=0)
        else:
            state['job']['empty_observations'] = min(3, state['job']['empty_observations'] + 1)
            program.status = 'uncertain'
    side = 'left' if request['side'] == 'auto' else request['side']
    return RigidClock(api, actual_request(request, side), current, [own_opening(api, side)] * 2, [2], reviewer=completed)
