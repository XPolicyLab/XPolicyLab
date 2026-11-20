"""Hold current public targets for an exact number of official command ticks."""


def run(actions, *, emit_one, read_count):
    if type(actions) is not int or not 0 <= actions <= 1600:
        raise ValueError('A bounded official action count is required')
    if not callable(emit_one) or not callable(read_count):
        raise TypeError('Explicit public tick emitter and own command counter required')
    for _ in range(actions):
        before = read_count()
        emit_one()
        if read_count() != before + 1:
            raise RuntimeError('Official hold emitter must consume exactly one command tick')
    return actions
