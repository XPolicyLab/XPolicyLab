import numpy as np
import torch
import re

from dexbotic.model.dexbotic_arch import (ActionOutputForCausalLM,
                                          DexboticConfig, DexboticForCausalLM)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.conversation import KeywordsStoppingCriteria



class DiscreteVLAForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = DexboticConfig

    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        attempt = 0
        while attempt < 40:
            try:
                return self._real_inference_action(input_ids, image_tensor, inference_args, **kwargs)
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {e}")

    def _real_inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        conv = inference_args.get('conv')
        tokenizer = inference_args.get('tokenizer')
        vocab_size = inference_args.get('vocab_size')
        action_norms = inference_args.get('action_norms')

        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            outputs = self.generate(input_ids,
                                    images=image_tensor,
                                    max_new_tokens=1024,
                                    do_sample=True,
                                    temperature=0.7,
                                    return_dict_in_generate=True,
                                    stopping_criteria=[stopping_criteria])

            outputs = outputs.sequences[0, input_ids.shape[1]:]
            outputs = tokenizer.decode(outputs, skip_special_tokens=False)
            outputs = outputs.strip(stop_str)
        actions = self._discrete_action_to_continuous(outputs, vocab_size)
        actions = self._denorm(actions, action_norms)
        actions = actions.tolist()

        return actions
    
    def _discrete_action_to_continuous(self, action_str: str, vocab_size: int):
        """ Discrete action [0, 1, 2, ..., vocab_size-1] to continuous action [-1, 1]
        """
        actions = re.findall(r'\d+', action_str)[:7]
        actions = np.array([int(action) for action in actions], dtype=np.float32).reshape(1, -1)
        actions = (actions / (vocab_size - 1)) * 2 - 1
        return actions