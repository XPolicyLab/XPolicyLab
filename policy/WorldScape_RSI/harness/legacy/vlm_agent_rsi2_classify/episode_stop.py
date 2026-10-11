"""A failure-only stop request, separate from native task success."""
def failure_stop_reason(response):
    if response.get('abort') is not True:
        return None
    if response.get('expected_outcome') != 'failure':
        raise ValueError('Abort cannot assert native success')
    if response.get('commands') not in ([], None):
        raise ValueError('Abort must not include robot commands')
    reason = response.get('abort_reason')
    if not isinstance(reason, str) or len(reason.strip()) < 15:
        raise ValueError('Abort requires an explicit reason')
    return reason.strip()
