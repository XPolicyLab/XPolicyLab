prompting_templates= {
  "v0": "<image>\nWhere should i move hand to: {} ? A: ",
  "v1": "Imagine you are a robot programmed for manipulation tasks. Your assigned task is {}. <image>\n Analyze this series of images to decide your next move, which could involve moving your hand to different locations and open or close your hand for manipulation.",
}

prompting_templates_empty = {
  "v0": "<image>\nWhere should i move hand to: finish the task ? A: ",
  "v1": "Imagine you are a robot programmed for manipulation tasks. Your assigned task is finish the task. <image>\n Analyze this series of images to decide your next move, which could involve moving your hand to different locations and open or close your hand for manipulation.",
}

def preprocess_language_instruction(
  language_label, valid_his_len, data_args
):

  language_instruction = prompting_templates[data_args.prompt_version].format(language_label)

  if data_args.ignore_language:
    language_instruction = prompting_templates_empty[data_args.prompt_version]

  if data_args.add_his_imgs:
    history_prompt_text = "You have been given a video of history observations:" + \
      "<image>\n" * valid_his_len
    history_prompt_text += "and current observation:<image>\n"
    language_instruction = language_instruction.replace(
      "<image>\n",
      history_prompt_text
    )
  return language_instruction
