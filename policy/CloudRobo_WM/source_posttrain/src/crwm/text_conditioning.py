import copy
import hashlib
import json
import warnings
MODEL_ID = 'Wan-AI/Wan2.2-TI2V-5B'
TOKENIZER_ID = 'Wan-AI/Wan2.1-T2V-1.3B'
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

def text_contract(context_len=128, model_id=MODEL_ID, tokenizer_model_id=TOKENIZER_ID):
    if type(context_len) is not int or context_len < 1:
        raise ValueError('Text context_len must be a positive integer')
    return dict(version='crwm.text.v1', context_len=context_len, model_id=str(model_id), encoder='wan_video_text_encoder', tokenizer_model_id=str(tokenizer_model_id), tokenizer_subfolder='google/umt5-xxl', clean='whitespace', add_special_tokens=True, padding_side='right', padding='max_length', dit_padding='masked', overflow='error')

def validate_text_contract(value):
    if not isinstance(value, dict):
        raise ValueError('Missing or invalid text conditioning contract')
    expected = text_contract(value.get('context_len'), value.get('model_id'), value.get('tokenizer_model_id'))
    if value != expected or not value['model_id'] or (not value['tokenizer_model_id']):
        raise ValueError('Unsupported text conditioning contract: padding/tokenizer/encoder/overflow differs')
    return value

def checkpoint_contract(value, *, warn_legacy=False):
    if not isinstance(value, dict):
        return value
    result = copy.deepcopy(value)
    if 'text' not in result:
        if warn_legacy:
            warnings.warn('Legacy S96/A90 checkpoint has no text metadata; interpreting it as the historical masked 128-token Wan text contract.', UserWarning, stacklevel=2)
        result['text'] = text_contract()
    validate_text_contract(result['text'])
    return result

def cache_filename(prompt, contract):
    validate_text_contract(contract)
    encoder_id = 'wan22ti2v5b'
    if contract['model_id'] != MODEL_ID or contract['tokenizer_model_id'] != TOKENIZER_ID:
        encoder_id = 'wantext_' + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16]
    return hashlib.sha256(prompt.encode()).hexdigest() + f".t5_len{contract['context_len']}.{encoder_id}.pt"

def check_prompt_lengths(tokenizer, prompts, contract):
    lengths = tokenizer.token_lengths(prompts, add_special_tokens=contract['add_special_tokens'])
    if any((n > contract['context_len'] for n in lengths)):
        raise ValueError(f"Text exceeds context_len={contract['context_len']}: longest={max(lengths)} tokens (including prefix/special tokens). Refusing silent truncation; increase the shared text length and rebuild caches.")
    if any((n < 1 for n in lengths)):
        raise ValueError('Text encoder requires at least one valid token')
    if tokenizer.tokenizer.padding_side != contract['padding_side']:
        raise ValueError('Tokenizer padding_side differs from the text contract')
    return lengths

def cached_context(payload, prompt, contract, *, allow_legacy=True, warn_legacy=True):
    import torch
    validate_text_contract(contract)
    metadata = payload.get('text_contract')
    if metadata is None:
        if not allow_legacy or contract != text_contract():
            raise ValueError('Untagged text cache is only compatible with the historical 128-token Wan default; rebuild it')
        if warn_legacy:
            warnings.warn('Legacy text cache has no encoder/tokenizer contract. Accepted only for the historical 128-token Wan default; validate lengths and cache metadata before use.', UserWarning, stacklevel=2)
    elif validate_text_contract(metadata) != contract:
        raise ValueError('Text cache contract mismatch (length, padding, tokenizer or encoder)')
    context, mask = (payload['context'], payload['mask'])
    if not torch.is_tensor(context) or not torch.is_tensor(mask) or context.ndim != 2 or (context.shape[0] != contract['context_len']) or (mask.shape != (contract['context_len'],)):
        raise ValueError('Text cache tensor shape differs from its contract')
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Text validity mask must be binary')
    mask = mask.bool()
    if not mask.any() or (~mask[:-1] & mask[1:]).any():
        raise ValueError('Text cache must contain valid tokens followed by right padding')
    if not torch.isfinite(context[mask]).all():
        raise ValueError('Nonfinite valid text embeddings')
    if metadata is not None:
        if payload.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Text cache prompt identity mismatch')
        if payload.get('untruncated_length') != int(mask.sum()):
            raise ValueError('Text cache token count mismatch or truncated text')
    return (torch.where(mask[:, None], context, 0), mask)
