
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib

from typing import Dict, Optional, Sequence, List
from human_plan.vila_train.args import DataArguments

import transformers
import tokenizers

import torch

import numpy as np

from llava.mm_utils import tokenizer_image_token

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
  is_multimodal = data_args.is_multimodal
  if not is_multimodal:
    return sources

  for source in sources:
    for sentence in source:
      if DEFAULT_IMAGE_TOKEN in sentence['value']:
        sentence['value'] = sentence['value'].replace(
            DEFAULT_IMAGE_TOKEN, '').strip()
        sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
        sentence['value'] = sentence['value'].strip()
        if "mmtag" in conversation_lib.default_conversation.version:
          sentence['value'] = sentence['value'].replace(
              DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
      replace_token = DEFAULT_IMAGE_TOKEN
      if data_args.mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
      sentence["value"] = sentence["value"].replace(
          DEFAULT_IMAGE_TOKEN, replace_token)

  return sources

def preprocess_v1_one_round(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
  assert has_image
  conv = conversation_lib.default_conversation.copy()
  roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

  # Apply prompt templates
  conversations = []
  for i, source in enumerate(sources):
    if roles[source[0]["from"]] != conv.roles[0]:
      # Skip the first one if it is not from human
      source = source[1:]

    conv.messages = []
    for j, sentence in enumerate(source):
      role = roles[sentence["from"]]
      assert role == conv.roles[j % 2], f"{i}"
      conv.append_message(role, sentence["value"])
    conversations.append(conv.get_prompt())

  # Tokenize conversations
  input_ids = torch.stack([
      tokenizer_image_token(
          prompt, tokenizer, return_tensors='pt'
      ) for prompt in conversations
  ], dim=0)

  targets = input_ids.clone()

  assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

  # Mask targets
  sep = conv.sep + conv.roles[1] + ": "
  for conversation, target in zip(conversations, targets):
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    rounds = conversation.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for i, rou in enumerate(rounds):
      if rou == "":
        break

      parts = rou.split(sep)
      if len(parts) != 2:
        break
      parts[0] += sep

      round_len = len(tokenizer_image_token(rou, tokenizer))
      instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2

      if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
        round_len -= 1
        instruction_len -= 1

      target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

      cur_len += round_len
    target[cur_len:] = IGNORE_INDEX

    if cur_len < tokenizer.model_max_length:
      if cur_len != total_len:
        target[:] = IGNORE_INDEX
        print(
            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
            f" (ignored)"
        )

  return dict(
      input_ids=input_ids,
      labels=targets,
  )


from human_plan.utils.action_tokenizer import ActionTokenizer

from llava.mm_utils import tokenizer_image_token

def preprocess_multimodal_vla(
    language_label: str,
    data_args: DataArguments,
    rank=None
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return language_label

    assert DEFAULT_IMAGE_TOKEN in language_label

    sentence_chunks = [chunk.strip() for chunk in language_label.split(DEFAULT_IMAGE_TOKEN)]

    sentence_chunks = [
        chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
    ] + [sentence_chunks[-1]]

    language_label = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()


    replace_token = DEFAULT_IMAGE_TOKEN
    if "mmtag" in conversation_lib.default_conversation.version:
        replace_token = "<Image>" + replace_token + "</Image>"
    if data_args.mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    language_label = language_label.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return language_label


def preprocess_vla(
    language_label,
    raw_input,
    raw_label,
    label_mask,
    action_tokenizer: ActionTokenizer,
    tokenizer: transformers.PreTrainedTokenizer,
    mask_input=False,
    mask_ignore=False,
    raw_action_label=False,
    traj_action_output_dim=1,
    input_placeholder_diff_index=False,
    sep_query_token=False,
    language_response=None,
    include_response=False,
    include_repeat_instruction=False,
    raw_language_label=None
) -> Dict:

  conv = conversation_lib.default_conversation.copy()
  conv.messages = []
  conv.append_message(conv.roles[0], language_label)

  if language_response is None:
    language_response = ""

  if raw_action_label:

    raw_label = raw_label.reshape(-1, 46)
    label_mask_for_tokenizer = label_mask.reshape(-1, 46)[:, 0]
    label_mask = label_mask.reshape(-1, 46)
    if sep_query_token:
      # 10, 46, 1
      label_mask_for_tokenizer = label_mask_for_tokenizer.repeat(2)

    action_text = action_tokenizer(
      (np.ones_like(label_mask_for_tokenizer) * 0.5).reshape(-1),
      label_mask_for_tokenizer.reshape(-1)
    )
  else:
    label_mask_for_tokenizer = label_mask
    action_text = action_tokenizer(
      raw_label, label_mask_for_tokenizer
    )

  if include_response:
    action_text = language_response + action_text

  if include_repeat_instruction:
    assert raw_language_label is not None
    action_text = f"The current instruction is {raw_language_label}" + action_text   

  conv.append_message(conv.roles[1], action_text)

  assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

  input_ids = tokenizer_image_token(
    conv.get_prompt(), tokenizer, return_tensors='pt'
  ).reshape(-1)

  targets = input_ids.clone()
  response_tokenized_len = len(tokenizer_image_token(action_text, tokenizer))

  targets[: -(response_tokenized_len)] = IGNORE_INDEX
  # Do not predict stop sign
  targets[-1] = IGNORE_INDEX

  if mask_input:
    if input_placeholder_diff_index:
      placeholder_index = action_tokenizer.input_placeholder_token_idx - \
        torch.arange(int(np.prod(label_mask_for_tokenizer.shape)), dtype=input_ids.dtype, device=input_ids.device)
      input_ids[-(np.prod(label_mask_for_tokenizer.shape) + 1): -1] = placeholder_index
    else:
      input_ids[-(np.prod(label_mask_for_tokenizer.shape) + 1): -1] = action_tokenizer.input_placeholder_token_idx
  
  if mask_ignore:
    targets[targets == action_tokenizer.invalid_token_idx] = IGNORE_INDEX

  if raw_action_label:
    
    if input_placeholder_diff_index:
      placeholder_index = action_tokenizer.input_placeholder_token_idx - \
        torch.arange(int(np.prod(label_mask_for_tokenizer.shape)), dtype=input_ids.dtype, device=input_ids.device)
      targets[-(np.prod(label_mask_for_tokenizer.shape) + 1): -1] = placeholder_index
      # targets[-(np.prod(label_mask.shape) + 1): -1] = action_tokenizer.input_placeholder_token_idx
    else:
      targets[-(np.prod(label_mask_for_tokenizer.shape) + 1): -1] = action_tokenizer.input_placeholder_token_idx

  """
  Below
  """
  if raw_action_label:
    return dict(
        input_ids=input_ids,
        labels=targets,
        raw_action_label=raw_label,
        raw_action_mask=label_mask,
        proprio_input=raw_input,
    )
  else:
    return dict(
        input_ids=input_ids,
        labels=targets,
        proprio_input=raw_input,
    )

def preprocess_vla_qa(
    language_label,
    tokenizer: transformers.PreTrainedTokenizer,
    response_text: str,
) -> Dict:

  conv = conversation_lib.default_conversation.copy()
  conv.messages = []
  conv.append_message(conv.roles[0], language_label)

  conv.append_message(conv.roles[1], response_text)

  assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

  input_ids = tokenizer_image_token(
    conv.get_prompt(), tokenizer, return_tensors='pt'
  ).reshape(-1)

  targets = input_ids.clone()
  response_tokenized_len = len(tokenizer_image_token(response_text, tokenizer))

  targets[: -(response_tokenized_len)] = IGNORE_INDEX
  targets[-1] = IGNORE_INDEX
  # print(input_ids)
  # print(targets)
  """
  Below
  """
  return dict(
      input_ids=input_ids,
      labels=targets,
  )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
  """
  Given a list of sources, each is a conversation list. This transform:
  1. Add signal '### ' at the beginning each sentence, with end signal '\n';
  2. Concatenate conversations together;
  3. Tokenize the concatenated conversation;
  4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
  """
  # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
  #     return preprocess_plain(sources, tokenizer)
  # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
  #     return preprocess_llama_2(sources, tokenizer, has_image=has_image)
  if conversation_lib.default_conversation.version.startswith("v1"):
    # return preprocess_v1(sources, tokenizer, has_image=has_image)
    return preprocess_vla(sources, tokenizer, has_image=has_image)
  # if conversation_lib.default_conversation.version == "mpt":
  #     return preprocess_mpt(sources, tokenizer, has_image=has_image)
  # add end signal and concatenate together

  conversations = []
  for source in sources:
    header = f"{conversation_lib.default_conversation.system}\n\n"
    conversation = _add_speaker_and_signal(header, source)
    conversations.append(conversation)
  # tokenize conversations

  def get_tokenize_len(prompts):
    return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

  if has_image:
    input_ids = [tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt') for prompt in conversations]
  else:
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]

  targets = copy.deepcopy(input_ids)
  for target, source in zip(targets, sources):
    if has_image:
      tokenized_lens = get_tokenize_len(
          [header] + [s["value"] for s in source])
    else:
      tokenized_lens = _tokenize_fn(
          [header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
    speakers = [sentence["from"] for sentence in source]
    _mask_targets(target, tokenized_lens, speakers)

  return dict(input_ids=input_ids, labels=targets)
