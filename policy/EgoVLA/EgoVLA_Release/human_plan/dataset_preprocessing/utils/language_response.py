
language_response_candidate_list = [
  "The future hand trajectory is ",
  "It is ",
  "The future hand trajectory in 2D and 3D space of this person's is ",
  "Sure, it is ",
  "Sure, the future hand trajectory is ",
  " ",
  "The 2D / 3D hand trajectory of the person is ",
  "Sure, "
]


import random

random.seed(7105)


def sample_text_response():
  text_response = random.sample(
    language_response_candidate_list, 1
  )[0]
  return text_response