import math
from typing import List, Any

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def split_train_eval(all_seqs: List[Any], split_skip: int=10):
  train_seqs = []
  eval_seqs = []
  for idx, sample in enumerate(all_seqs):
    if idx % split_skip == 0:
      eval_seqs.append(sample)
    else:
      train_seqs.append(sample)
  return train_seqs, eval_seqs
