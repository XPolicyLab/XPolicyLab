"""Class for doing prompting/other data formatting for tasks

For example, converting points to text, or applying prompt templates
"""
import dataclasses
import re
from collections import Counter
from typing import Optional, Dict, Tuple

import numpy as np

GENERAL_PROMPTS_V1 = {
    "short_answer": [
        "Answer this question very briefly\n{question}",
        "{question} Answer with a few words",
        "{question} Response very briefly",
        "{question} Answer directly without any details, explanation, or elaboration",
        "I have a question about this image, please answer it very briefly: {question}",
        "Question: {question} Short Answer:",
        "Question: {question}\nShort Answer:",
        '{question}\nAnswer the question as briefly as possible.',
        'Answer very briefly:\n{question}',
        'The question "{question}" can be answered using the image. A short answer is',
        "{question} Based on the image, respond to this question with a short answer:",
        "{question} Short answer:",
        "{question} A short answer to the question is",
        "Give a short, matter-of-fact answer to this question: {question}",
        "Give me a simple, direct answer to this question, do not elaborate or explain your answer:\n{question}"
    ],
    "short_caption": [
        'Caption the image with 1 or two sentences',
        'Write a very short description of this image.',
        'Briefly describe the image.',
        'Look and this image, and then summarize it in a sentence or two.',
        'Write a brief caption describing the image',
        'Brief Caption:'
        'A short image caption:',
        'A short image description',
        'Briefly describe the content of the image.',
        'Can you give me one sentence summary of the picture?',
        'How would you describe this image in a sentence or two?',
    ],
    "long_caption": [
        'Describe this image.',
        'Describe this image',
        'describe the image',
        'Write a long description of this image.',
        'caption the picture',
        'Caption',
        'caption',
        'Construct a long caption for this image',
        'Generate a caption',
        'Create a detailed caption',
        'Write a long caption',
        'Describe this image in detail',
        'Describe this',
        'describe this',
        'Caption this',
        'What can be seen in this image?',
        'What do you see in the image?',
        'Look at this photo carefully and then tell me about it in detail',
        'Write a long description of this image',
        'Tell me about this picture.',
        'Write a paragraph about this image.',
        'Look at this image carefully and then describe it in detail',
        'Generate a long caption about this image.'
    ],
    "long_caption_no_pointing": [
        'Describe this image in detail, but without any pointing.',
        'Write a long description of this image, do not produce any points.',
        'Tell me about this picture, use plain text only.',
        'Generate a plain text description of this caption',
        "What is in this image?\nNo pointing\nGive lots of detail"
        "Write a long caption.\nDo not use image coordinates\nOutput a full paragraph"
    ],
    "transcript": [
        'Describe this image as if you are a person speaking',
        'Imagine you are a person talking about this image. Generate a transcript of what you would say.',
        "Generate an audio transcript of a person describing this image",
        "Create a transcript of a human describing this image out load",
        "Describe this in this style of a human talking",
    ],
    "refexp_pointing": [
        'Where is the \"{refexp}\"?',
        'Point to {refexp}',
        'point at {refexp}',
        'Find the {refexp}.',
        'Which object in the image does \"{refexp}\" refer to?',
        'Locate the object \"{refexp}\" refers to.',
        'Point to the object that best matches the expression:\n{refexp}\n',
        'What object could be described as: {refexp}.\nPoint:',
        'Referring Expression: {refexp}.\nPoint:',
        'Expression: {refexp}\nPoint to the refexp',
        'Task: Point to the object that best matches the expression.\nExpression: {refexp}\nPoint:',
        'Instruction: Locate the object that matches the expression by returning a point.\nReferring Expression: {refexp}\n',
        'Help me find an object in this image by pointing to the {refexp}',
        'What point of the image might the expression \'{refexp}\' refer to?',
    ],
    "plain": ["{question}"],
    "multiple_choice": [
        "{question}\n{options}\nReturn only the letter of the best answer option",
        "Answer this question by naming one of the provided options:\n{question}\n{options}",
        "{question}\n{options}\nWhat option best answers the question?",
        "{question}\n{options}\nReturn the best answer option",
        "Look at the options, then return the letter of the option that best answers the question.\nQuesiton: {question}\nOptions: {options}",
        "{question}? Select an answer option from:\n{options}",
        "{question}\nSelect an answer option from:\n{options}\n\n",
        "Question: {question}? Options: {options} Answer:",
        "Answer the question by selecting an answer options\nQuestion: {question}\nOptions: {options}",
        "{question}?\n{options}\nReturn only the letter of the correct answer",
        "Help me answer this question: \"{question}\", by stating which of the following options is correct\n{options}."
    ],
    "pointing": [
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to the {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
    ],
    "point_count": [
        "How many {label} are there?",
        "How many {label}?",
        "How many {label}.",
        "how many {label}.",
        "how many {label}?",
        "How many {label} are there in the image?",
        "Tell me how many {label} there are",
        "Tell me how many {label} there are and point to them.",
        "how many {label}",
        "Tell me where each {label} is.",
        "Tell me how many {label} are in the image",
        "count {label}",
        "count every {label}",
        "count each {label}",
        "count {label}.",
        "Count the {label}.",
        "How many {label} do you see?",
        "How many {label} are visible?",
        "Count all the {label}",
        "how mmny {label}?",
        "Count every {label} in the picture.",
        "Count all the {label}",
        "Count each {label}",
        "Point to and count the {label} in the picture.",
        "Point and count {label}",
        "Point to every {label}",
        "Locate the {label} and count them",
        "Locate every {label} and count them",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
        "Point at {label} and then tell me the count.",
        "What is the total number of {label} in the image?",
        "In all the picture, how many {label} are there?",
        "Point at the {label} and then count them.",
        "Point to all the visible {label} output the total count.",
        "Point to all the {label} visible and output the total count. \nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" and output the total count.",
        "Show me where the {label} are and output the total count.",
        "Where are the {label}? How many are there?",
        "Generate list of points showing where the {label} are and output the total count.",
        "Object: {label}\nInstruction: Point to the object and output the total count.",
        "find any {label} in the picture and output the total count.",
        "Can you see any {label} in the image? Point to them and output the total count.",
        "Can you point out all {label} in this image? How many are there?",
        "If there are any {label} present, indicate their positions and output the total count.",
        "How many {label} are there in the image? Point to them and output the total count.",
        "How many {label} are there in the image?",
        "Give me the count of {label} in the image.",
        "How many {label} are visible in the image?",
        "How many {label} are there?",
        "In the image, how many {label} are there?",
        "Can you count the number of {label} in the image?",
        "Can you count every {label} in the picture?",
        "Can you see any {label} in the image? How many are there?",
        "Are there any {label} in the image? How many are there?",
        "If you see any {label} in the image, give me the count. Otherwise, say 'This isn't in the image.'",
        "Object: {label}\nInstruction: How many are there?",
    ],
    "count_then_point": [
        "Count the {label} in the image, then point to them.",
        "How many {label} are there? Point to them.",
        "Count every {label} in the picture, then point to them.",
        "Locate the {label} and count them, then point to them.",
        "Find all the {label}. How many are there? Point to them.",
        "Find each {label}. How many are there? Point to them.",
        "Point to and count the {label} in the picture.",
    ],
    "only_count": [
        "Count the {label} in the image.",
        "How many {label} are there?",
        "Count every {label} in the picture.",
        "Locate the {label} and count them.",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
    ],
}


STYLE_TO_GENERAL_PROMPT = {
    "vqa2": "short_answer",
    "coco_captioning": "short_caption",
    "gqa": "short_answer",
    "ocr_vqa": "short_answer",
    "tally_qa": "short_answer",
    "text_vqa": "short_answer",
    "okvqa": "short_answer",
    "chart_qa": "short_answer",
    "doc_qa": "short_answer",
    "info_qa": "short_answer",
    "science_qa": "multiple_choice",
    "ai2_diagram": "multiple_choice",
    "a_okvqa_mc": "multiple_choice",
    "a_okvqa_da": "short_answer",
    "long_caption": "long_caption",
    "scifi_charts": "short_answer",
    "scifi_charts_qa": "short_answer",
    "pointing": "pointing",
    "point_count": "point_count",
    "count_then_point": "count_then_point",
    "only_count": "only_count",
    "plain": "plain",
}


def apply_keywords(prompt, example, keywords):
    for keyword in keywords:
        res = prompt.split("{"+keyword+"}", maxsplit=2)
        prompt = res[0] + example[keyword] + res[1]
    return prompt


def apply_keyword_prompt(prompts, example, rng, keywords=None, dbg=False):
    if isinstance(prompts, list):
        assert keywords is None
        all_keywords = [sorted(re.findall("{([^{}]+)}", x)) for x in prompts]
        keywords = all_keywords[0]
        assert len(keywords) == len(set(keywords)), f"Repeated keywords in {keywords}"
        assert all(keywords == x for x in all_keywords), f"Inconsistent keywords in prompts {all_keywords}"
        assert not any("{" not in word[1:-1] and "}" in word[1:-1] for word in keywords)

        for k in keywords:
            assert k in example, f"Example missing expected field {k}, example={example}"

    if dbg:
        prompt = prompts[0]
    else:
        prompt = prompts[rng.randint(0, len(prompts))]
    return apply_keywords(prompt, example, keywords)


DEMO_STYLES = [
    "action_planning",
    "affordance_detection",
    "point_count",
    "pointing",
    "user_qa",
    "long_caption",
]


@dataclasses.dataclass
class DataFormatter:
    """Applies prompt templates and adds system prompts to construct text inputs/output"""
    prompt_templates: str = "none"  # How to template prompts for examples
    message_format: str = "none"  # How to format messages
    system_prompt: Optional[str] = None  # How to generate system prompts
    always_start_with_space: bool = False  # Always include a leading space for the first bit of text
    default_inference_len: int = 65  # Inference len for length-conditioned prompting
    select_answer: str = "best"  # How to select answer for questions with many answers
    debug: bool = False  # deterministic mode for debugging

    def points_to_text(self, points, scale, label_text, alt_text):
        if isinstance(scale, (tuple, list)):
            points /= np.array(scale)[None, :]
        else:
            points *= (100/scale)
        points = [[round(x, 1), round(y, 1)] for x, y in points]
        points.sort(key=lambda x: x[0]*10000 + x[1])
        if len(points) == 1:
            x_str, y_str = points[0]
            return f"<point x=\"{x_str:0.1f}\" y=\"{y_str:0.1f}\" alt=\"{alt_text}\">{label_text}</point>"
        point_text = []
        for ix, (x, y) in enumerate(points, start=1):
            point_text.append(f"x{ix}=\"{x:0.1f}\"")
            point_text.append(f"y{ix}=\"{y:0.1f}\"")
        point_text = " ".join(point_text)
        return f"<points {point_text} alt=\"{alt_text}\">{label_text}</points>"

    def format_annotated_text(self, answer, point_annotations):
        for point_annotation in point_annotations:
            parts = answer.split("<|POINT|>", maxsplit=1)
            point_text = self.points_to_text(
                np.array(point_annotation["points"]), 100,
                point_annotation["inline_text"], point_annotation["alt_text"])
            answer = parts[0] + point_text + parts[1]
        return answer

    def format_points(self, example):
        if "points" not in example:
            return None
        points = example["points"]
        style = example["style"]
        if "label" in example:
            label = example["label"].lower()
        else:
            label = example["question"]
        if len(points) == 0:
            if style in ["pointing", "point_count"]:
                return "There are none."
            else:
                raise NotImplementedError()
        if "point_scale" in example:
            # Points are already normalized
            point_txt = self.points_to_text(points, example["point_scale"], label, label)
        else:
            # Points are in pixel coordinate
            h, w = example["image"].shape[:2]
            point_txt = self.points_to_text(points, [w/100, h/100], label, label)

        if style == "point_count":
            return f"Counting the {point_txt} shows a total of {len(points)}."
        else:
            return point_txt

    def format_options(self, example):
        if "options" in example:
            prefixes = "abcdefg".upper()
            options = example["options"]
            option_text = "\n".join(f"{prefix}: {opt}" for prefix, opt in zip(prefixes, options))
            option_names = prefixes[:len(options)]
        else:
            options = example["unlabelled_options"]
            option_text = "\n".join(options)
            prefixes = options
            option_names = options
        if "answer_idx" in example:
            output = prefixes[example["answer_idx"]]
        else:
            output = None
        return output, example["question"] + "\n" + option_text + "\n", dict(option_names=option_names)

    def select_vqa_answer(self, answers, rng):
        if answers is None or isinstance(answers, str):
            return answers
        if self.select_answer == "first":
            return min(answers)
        if self.select_answer == "best":
            counts = Counter(answers)
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            return candidates[rng.randint(0, len(candidates))]
        else:
            raise NotImplementedError(self.select_answer)

    def format_messages(self, messages):
        """Applies system formatting to ith message from a sequence of messages"""
        out = []
        # if massage is a string list and message_format is role, 
        # add "User:" and "Assistant:" to the messages
        for ix, message in enumerate(messages):
            is_user = ix % 2 == 0
            if self.message_format == "none" or self.message_format is None:
                pass
            elif self.message_format == "role":
                if is_user:
                    message = "User: " + message + " Assistant:"
            else:
                raise NotImplementedError(self.message_format)

            if ix != 0 or self.always_start_with_space:
                message = " " + message
            out.append(message)
        return out

    def get_system_prompt(self, style, for_inference, messages, rng):

        # For eval only dataset
        if style == "eval_short_answer":
            style = "vqa2"
        elif style == "eval_multiple_choice":
            style = "a_okvqa_mc"

        if self.system_prompt == "style":
            return style + ":"

        # setting in train_multitask_model.py 
        elif self.system_prompt == "demo_or_style":
            if style == "android_control" or style == "demo":
                # android is a special case since I hacked in prefix in the preprocessor
                prefix = ""
            elif style == "action":
                prefix = ""
            elif style in DEMO_STYLES and rng.random() > 0.1 and not self.debug:
                # Use style prompt 10% of the time so we can still get task-specific output
                prefix = ""
            else:
                prefix = style + ":"

        elif for_inference and self.system_prompt == "style_and_length":
            v2 = self.system_prompt == "style_and_length_v2"
            inference_len = self.default_inference_len
            n = None if inference_len is None else str(inference_len)
            if n is not None and len(n) > 0:  # allow empty string to signal unconditioned
                prefix = style + " " + n + ":"
            else:
                prefix = style + " :"  # FIXME remove the space
        elif self.system_prompt == "style_and_length":
            std = 25
            if rng.random() > 0.10:
                n = len(messages[-1])
                n += int(rng.normal(scale=std))
                prefix = style + " " + str(n//15) + ":"
            else:
                prefix = style + " :"  # FIXME remove the space
        else:
            raise NotImplementedError(self.system_prompt)

        return prefix

    def get_user_prompt(self, example, is_training=True, for_inference=False, rng=None):
        """Build a list of strings of what a user might type in to the model for the given example,
        and its responses, by applying a prompt template to the fields in `example`

        Uses the `style` field to understand what the task/output style is
        """
        # if no "messages" or "message_list" key
        # or if they aren't a string or list
        style = example.get("style")
        output = None
        metadata = None

        if "prompt" in example:
            # Examples have a complete user prompt pre-specified, usually for eval sets
            prompt = example["prompt"]

        elif self.prompt_templates == "none":
            # Bare-bone prompt with no templating or instructions
            # handle with pointing and point_count style
            if style in ["pointing", "point_count"]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    if "label" in example:
                        prompt = example["label"]
                        prompt = prompt.lower()
                    else:
                        prompt = example["label_cased"]
                output = self.format_points(example)
            
            # handle multiple choice
            elif "question" in example and ("options" in example or "unlabelled_options" in example):
                output, prompt, metadata = self.format_options(example)
            
            elif "question" in example:
                prompt = example["question"]
            else:
                prompt = ""

        elif self.prompt_templates == "uber_model":
            if not isinstance(style, str):
                assert style in ["ai2_diagram_no_letter", "ai2_diagram" ]
                output, prompt, metadata = self.format_options(example)
            else:
                # We template long captions and pointing since they are "demo" tasks, and use
                # plain text for everything else
                if style == "long_caption":
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["long_caption"], example, rng, dbg=self.debug)
                elif style in ["pointing", "point_count"]:
                    # output, prompt, metadata = self.format_points(example)
                    if "question" in example:
                        prompt = example["question"]
                    else:
                        if "label" in example:
                            prompt = example["label"].lower()
                        else:
                            prompt = example["label_cased"]
                        prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], dict(example, label=prompt), rng, dbg=self.debug)
                    output = self.format_points(example)
                elif "prompt" in example:
                    prompt = example["prompt"]
                elif "question" in example and ("options" in example or "unlabelled_options" in example):
                    output, prompt, metadata = self.format_options(example)
                elif "question" in example:
                    prompt = example["question"]
                else:
                    prompt = ""
        else:
            raise NotImplementedError(self.prompt_templates)

        # expect "answers", "answer", "answer_with_points", "text" in example
        if output is None and not for_inference:
            # for multiple string outputs, select one.
            if "answers" in example:
                output = self.select_vqa_answer(example["answers"], rng)
            elif "answer" in example:
                output = example["answer"]
                if "answer_annotations" in example:
                    output = self.format_annotated_text(output, example["answer_annotations"])
            elif "answer_with_points" in example:
                output = example["answer_with_points"]
            elif "text" in example:
                output = example["text"]
            else:
                print(example)
                raise ValueError("No output in example, if this is an inference-only task make sure `for_inference` is True")

        return prompt, output, metadata

    def _format_example(self, message, example, is_training, for_inference, rng):
        metadata = {}
        for k in ["answer_idx", "answers", "answer", "points", "options"]:
            if k in message:
                metadata[k] = message[k]
        
        # for a dataset, in each example(dict) check if has "message_list" or "messages" key 
        # if so check if it is a string or list
        if isinstance(message, str):
            messages = [message]
        elif isinstance(message, list):
            messages = message
        
        # assume that message is a dict
        elif "messages" in message:
            # Example directly contains the prompts/message to use
            messages = message["messages"]

        # "message_list" or "messages" isn't a string or list
        # or example doesn't contain them
        elif isinstance(message, dict):
            # An example that requires a custom prompt
            # treat it as it does not have "messages" and "message" key
            prompt, response, extra_metadata = self.get_user_prompt(message, is_training, for_inference=for_inference, rng=rng)
            if extra_metadata:
                metadata.update(extra_metadata)
            if not for_inference:
                assert response is not None
                # for "message_list" if it is a list within dict, molmo \
                # does not support multiple conversation turns
                messages = [prompt, response]
            else:
                messages = [prompt]
        else:
            raise ValueError(f"Example type {type(message)} not understood")

        # Add the system prompt
        # now the messages is a list of strings.
        # normally, system_prompt not none 
        if self.system_prompt != None and self.system_prompt != "none":
            style = None
            if isinstance(message, dict):
                if "style" in message:
                    style = message.get("style", None)
            if style is None:
                style = example.get("style", None)
            prefix = self.get_system_prompt(style, for_inference, messages, rng=rng)
            # messages is a list of strings, and the first one is the system prompt
            if len(prefix) > 0 and len(messages[0]) > 0:
    
                with_system_prompt = prefix + " " + messages[0]
            elif len(prefix) > 0:
                with_system_prompt = prefix
            else:
                with_system_prompt = messages[0]
            messages = [with_system_prompt] + messages[1:]

        # Add the role annotations such as "User:" and "Assistant:"
        messages = self.format_messages(messages)
        return messages, metadata

    # Returns a formatted example and example metadata
    # Does not support returning metadata for "message_list", which is fine since we are not doing inference
    # if "message_list" has only one conversation (include one turn), it can be evaluated.
    def __call__(self, ex: Dict, is_training, for_inference, rng) -> Tuple[Dict, Dict]:
        if "message_list" in ex: 
            # if multiple conversations, will return string lists within a list
            return [self._format_example(msg, ex, is_training, for_inference, rng)[0]
                    for msg in ex["message_list"]], None
        elif "messages" in ex:
            return self._format_example(ex["messages"], ex, is_training, for_inference, rng)
        else:
            return self._format_example(ex, ex, is_training, for_inference, rng)
