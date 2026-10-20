import copy
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
    value = copy.deepcopy(value)
    expected = text_contract(value.get('context_len'), value.get('model_id'), value.get('tokenizer_model_id'))
    if value != expected or not value['model_id'] or (not value['tokenizer_model_id']):
        raise ValueError('Unsupported text conditioning contract: padding/tokenizer/encoder/overflow differs')
    return value

def checkpoint_contract(value):
    if not isinstance(value, dict) or 'text' not in value:
        raise ValueError('Missing CR-WM conditioning contract')
    result = copy.deepcopy(value)
    result['text'] = validate_text_contract(result['text'])
    return result

def check_prompt_lengths(tokenizer, prompts, contract):
    lengths = tokenizer.token_lengths(prompts, add_special_tokens=contract['add_special_tokens'])
    if any((n > contract['context_len'] for n in lengths)):
        raise ValueError(f"Text exceeds context_len={contract['context_len']}: longest={max(lengths)} tokens (including prefix/special tokens). Refusing silent truncation; increase the shared text length and rebuild caches.")
    if any((n < 1 for n in lengths)):
        raise ValueError('Text encoder requires at least one valid token')
    if tokenizer.tokenizer.padding_side != contract['padding_side']:
        raise ValueError('Tokenizer padding_side differs from the text contract')
    return lengths
