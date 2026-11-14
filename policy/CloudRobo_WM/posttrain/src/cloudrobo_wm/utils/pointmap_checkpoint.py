from cloudrobo_wm.utils.logging_config import get_logger
logger = get_logger(__name__)

def validate_pointmap_encoding(payload: dict, expected: dict, *, strict_shape: bool) -> None:
    saved = {'pointmap_encoding': payload['pointmap_encoding']} if 'pointmap_encoding' in payload else {}
    if saved == expected:
        return
    if strict_shape:
        raise ValueError(f'Strict point-map encoding metadata mismatch (saved, expected): ({saved}, {expected}). Use strict_shape=False only for a weight warm-start into a new experiment; the encoding cannot change on resume.')
    logger.warning('Non-strict point-map warm-start changes the input representation (saved=%s, destination=%s). Loading checkpoint weights; keeping the destination point-map encoding. This is not a strict resume.', saved, expected)
