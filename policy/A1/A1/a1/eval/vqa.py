"""Various VQA metrics used by different datasets"""
import re
import string
from collections import Counter
from typing import Optional, List
import logging
import editdistance
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from math_verify import parse, verify
from collections import deque
from a1.eval import mmmu_eval_utils, math_vista_utils

log = logging.getLogger(__name__)

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
    "youll": "you'll", "youre": "you're", "youve": "you've"}

manualMap = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles = ['a','an','the']

punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!']

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",outText,re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(ans, cache={}):
    if ans in cache:
        return cache[ans]
    ans = ans.replace('\n', ' ')
    ans = ans.replace('\t',' ')
    ans = ans.lower().strip()
    preprocessed = processDigitArticle(processPunctuation(ans))
    cache[ans] = preprocessed
    return preprocessed


def vqa_score(target, pred):
    """
    Evaluation with VQA 2 style preprocessing
    """
    pred = preprocess_answer(pred)
    if isinstance(target, list):
        target = Counter(preprocess_answer(x) for x in target)
        return min(target[pred] / 3.0, 1)
    else:
        return float(pred == target)

def robovqa_score(target, pred):
    pred = pred.replace("\n", "").lower()
    scores, bleu1s, bleu2s, bleu3s, bleu4s = 0, 0, 0, 0, 0
    for gt in target:
        gt = gt.replace("\n", "").lower()
        # log.info(f'pred: {pred}, gt: {gt}')
        if gt in ['yes', 'no']:
            pred = re.sub(r'\b\w*yes\w*\b', 'yes', pred)
            pred = re.sub(r'\b\w*no\w*\b', 'no', pred)
        score, bleu1, bleu2, bleu3, bleu4 = get_bleu_score(pred, gt)
        scores += score
        bleu1s += bleu1
        bleu2s += bleu2
        bleu3s += bleu3
        bleu4s += bleu4 
    return bleu1s/len(target)

def get_bleu_score(prediction, target):
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    candidate = list(prediction.split(" "))
    reference = [list(target.split(" "))]
    if target is not None:
        # print(f"pred:{pred}, gt:{gt}, bleu:{sentence_bleu(reference, candidate)}")
        if len(reference[0]) <= 1:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
        elif len(reference[0]) == 2:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
        elif len(reference[0]) == 3:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
        else:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
               
    score = (bleu1 + bleu2 + bleu3 + bleu4) / 4
    return score, bleu1, bleu2, bleu3, bleu4

def clevr_score(answers, pred):
    if not answers:
        return 0.0
    # For test, we use Reason-RFT's answer format, which is a single answer.
    solution = answers[0]
    reward = 0.0
    # match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
    # In clevrmath and superclevr data-preprocessing, the answer is wrapped in <CONCLUSION> tags,
    # we assume that the pred will show its answer in same way.
    match = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", pred, re.DOTALL)
    content_to_check = ""
    if match:
        content_to_check = match.group(1).strip()
    else:
        content_to_check = pred.strip()
    try:
        if solution == content_to_check:
            reward = 1.0
        elif float(verify(parse(content_to_check), parse(solution))) > 0:
            reward = 1.0
    except Exception:
        reward = 0.0
    return reward

def superclevr_score(answers, pred):
    return clevr_score(answers, pred)

def clevrmath_score(answers, pred):
    return clevr_score(answers, pred)

def extract_structured_info_from_sentence(sentence: str) -> tuple | None:
    """
    Args:
        str(sentence): Sentence that contains strctured information.
    Returns:
        tuple(function, object_description, new_value)
    """
    sentence = sentence.strip()
    
    # Match attribute (color, material, shape, size)
    # e.g., "The large glass cylinder's color changed from gray to purple."
    match = re.match(r"The (.*?)'s (color|material|shape|size) changed from .*? to (.*?)\.?$", sentence)
    if match:
        obj_desc = match.group(1).strip()
        attribute = match.group(2).strip()
        new_value = match.group(3).strip()
        func_name = f"change_{attribute}"
        return (func_name, obj_desc, new_value)

    # Match movement
    # e.g., "The small yellow glass cube moved from position [19, 6] to [29, 16]."
    match = re.match(r"The (.*?) moved from position .*? to (\[.*?\])\.?$", sentence)
    if match:
        obj_desc = match.group(1).strip()
        new_position = match.group(2).strip()
        return ('move', obj_desc, new_position)
        
    return None

def extract_items_from_nl(text: str) -> list[tuple]:
    """
    Extract each sentence in sentence.
    """
    if not text:
        return []
    if isinstance(text, list):
        text = ''.join(text)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    structured_list = []
    for sent in sentences:
        info = extract_structured_info_from_sentence(sent)
        if info:
            structured_list.append(info)
    return structured_list

def _calculate_trance_score(pred_list: list[tuple], sol_list: list[tuple]) -> float:
    '''
    Base on trance score function in Reson-RFT.
    '''
    reward = 0.0
    
    if not sol_list:
        return 0.0
    
    item_score = 1.0 / max(len(pred_list), len(sol_list)) if pred_list else 0
    
    pred_queue = deque(pred_list)
    # print(f"pred_queue: {pred_queue}")
    sol_queue = deque(sol_list)
    # print(f"sol_queue: {sol_queue}")
    
    # Full metch (func, object, value)
    full_mapping_num = 0
    exact_matches = [(p, s) for p in pred_queue for s in sol_queue if p == s]
    for p, s in exact_matches:
        if p in pred_queue and s in sol_queue:
            full_mapping_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += full_mapping_num * item_score

    # Partly match (func, object)
    partial_matches_1_num = 0
    partial_matches_1 = [(p, s) for p in pred_queue for s in sol_queue if p[:2] == s[:2]]
    for p, s in partial_matches_1:
        if p in pred_queue and s in sol_queue:
            partial_matches_1_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += partial_matches_1_num * item_score * 0.5
    
    # Partly match (func, value)
    partial_matches_2_num = 0
    partial_matches_2 = [(p, s) for p in pred_queue for s in sol_queue if (p[0], p[2]) == (s[0], s[2])]
    for p, s in partial_matches_2:
        if p in pred_queue and s in sol_queue:
            partial_matches_2_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += partial_matches_2_num * item_score * 0.5
    
    # Only func
    func_matches_num = 0
    func_matches = [(p, s) for p in pred_queue for s in sol_queue if p[0] == s[0]]
    for p, s in func_matches:
        if p in pred_queue and s in sol_queue:
            func_matches_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += func_matches_num * item_score * 0.25
    
    return reward

def trance_score(ans: str, pred: str) -> float:
    """
    Main function to calculate the trance score.
    """
    content_clean = pred.strip()
    
    # Parsing
    pred_list = extract_items_from_nl(content_clean)
    sol_list = extract_items_from_nl(ans)

    # Scoring
    return _calculate_trance_score(pred_list, sol_list)

def a_okvqa_score(target, pred):
    # A-OK-VQA eval scripts don't seem to do any answer pre-processing
    target = Counter([x.lower().strip() for x in target])
    return min(target[pred.lower().strip()] / 3.0, 1)


def select_mc_option(target, options):
    """
    Selects a multiple-choice option based on the model output

    The output is should exactly match one of the option, but contains
    some heuristic fallbacks in case the does not occur
    """
    target = target.lower().strip()
    n = len(options)
    options = [x.lower().strip() for x in options]
    assert len(set(options)) == n
    for ix, option in enumerate(options):
        if option == target:
            return ix

    contains = []
    for ix, option in enumerate(options):
        if target in option:
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]
    distances = [editdistance.eval(opt, target) for opt in options]
    return np.argmin(distances)


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
      target: Target string.
      prediction: Predicted string.
      theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
      ANLS score.
    """
    # Lowercase is not in https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
    # However https://rrc.cvc.uab.es/?ch=17&com=tasks says
    #  - "Answers are not case sensitive"
    #  - "Answers are space sensitive"
    edit_distance = editdistance.eval(target.lower(), prediction.lower())
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# From https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/main_parse_and_eval.py
def mmmu_score(
    target: List[str],
    response: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple-choice":
        options = metadata["options"]
        options = [opt for opt in options if len(opt) > 0]
        all_choices = [chr for chr in string.ascii_uppercase[:len(options)]]
        index2ans = {chr: option for chr, option in zip(all_choices, options)}
        parsed_pred = mmmu_eval_utils.parse_multi_choice_response(response, all_choices, index2ans)
        correct = mmmu_eval_utils.eval_multi_choice(target, parsed_pred)
    else: # open
        parsed_pred = mmmu_eval_utils.parse_open_response(response)
        correct = mmmu_eval_utils.eval_open(target, parsed_pred)
    return float(correct)


def real_world_qa_score(
    target: str,
    prediction: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple_choice":
        options = ["A", "B", "C", "D"]
        pred_idx = select_mc_option(prediction, options)
        gt_idx = options.index(target)
        score = pred_idx == gt_idx
    else:
        pred = preprocess_answer(prediction)
        gt = preprocess_answer(target)
        score = float(pred == gt)
    return score


def math_vista_score(
    response: str,
    metadata: dict,
    openai_api_key: str,
    use_api: bool = True,
):
    # extract answer using GPT-4.
    pid = metadata["example_id"]
    question_type = metadata["question_type"]
    answer_type = metadata["answer_type"]
    choices = metadata["choices"]
    target = metadata["answer"]
    query = metadata["query"]

    if use_api:
        extraction = math_vista_utils.extract_answer(
            pid, response, question_type, answer_type, choices, query, openai_api_key,
        )
    else:
        if question_type == "multi_choice":
            options = [chr(ord("A") + i) for i in range(len(choices))]
            pred_idx = select_mc_option(response, options)
            extraction = choices[pred_idx]
        else:
            if answer_type == "integer":
                try:
                    extraction = str(int(response))
                except:
                    extraction = response
            elif answer_type == "float":
                try:
                    extraction = str(float(response))
                except:
                    extraction = response
            else:
                extraction = response

    # calculate score
    precision = metadata["precision"]

    # normalize the extracted answer to match the answer type
    prediction = math_vista_utils.normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision,
    )

    # verify the prediction is true or false
    true_false = math_vista_utils.safe_equal(prediction, target)

    return true_false
