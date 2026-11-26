"""Read official episode instructions and extract explicit goals from them."""

import re


_INSTANCE_SECTION = '\n\n## Evaluator-supplied instance instruction\n\n'


def instruction_sources(obs):
    """Return the Detail Description and only the short text needed alongside it."""
    extra = obs.get('extra_info')
    detailed = extra.get('detailed_instruction') if isinstance(extra, dict) else None
    if not isinstance(detailed, str) or not detailed.strip():
        detailed = ''
    if detailed and 'read the current language instruction' not in ' '.join(detailed.casefold().split()):
        return detailed, ''
    for key in ('instruction', 'instructions'):
        short = obs.get(key)
        if isinstance(short, str) and short.strip():
            return detailed, short
    return detailed, ''


def episode_instruction(obs):
    """Keep the official text verbatim and label any delegated instance text separately."""
    detailed, short = instruction_sources(obs)
    return f'{detailed}{_INSTANCE_SECTION}{short}' if short and detailed else detailed or short


def _goal_text(instruction):
    """Parse concrete assignments separately from generic description and scoring prose."""
    return instruction.rsplit(_INSTANCE_SECTION, 1)[-1]


def normalize(text):
    return ' '.join(re.sub(r'(?<=\w)[\u2010-\u2014_-](?=\w)', ' ', text.lower()).split())


def basket_categories(instruction, categories):
    text = normalize(_goal_text(instruction))
    aliases = {normalize(category): category for category in categories}
    category_pattern = r'\b(' + '|'.join(re.escape(alias) for alias in aliases) + r')(?:es|s)?\b'
    sides = ('left', 'middle', 'right')
    for alias, side in (('center', 'middle'), ('centre', 'middle'), ('leftmost', 'left'), ('rightmost', 'right')):
        text = re.sub(rf'\b{alias}\b', side, text)
    if 'respectively' in text:
        found = re.findall(category_pattern, text)
        order = re.findall(r'\b(left|middle|right)\b', text)
        if len(found) == 3 and sorted(order) == sorted(sides) and len(set(found)) == 3:
            mapping = dict(zip(order, found))
            return [aliases[mapping[side]] for side in sides]
    mapping = {}
    for clause in re.split(r'[,;.!]|\band\b|\bthen\b', text):
        found = re.findall(category_pattern, clause)
        destinations = re.findall(r'\b(left|middle|right)\s+baskets?\b', clause)
        destinations += re.findall(r'\bbaskets?\s+(?:on|in|at)\s+(?:the\s+)?(left|middle|right)\b', clause)
        if not found and not destinations:
            continue
        if len(found) != 1 or len(destinations) != 1 or destinations[0] in mapping:
            raise ValueError(f'unclear basket assignment: {instruction!r}')
        mapping[destinations[0]] = aliases[found[0]]
    if set(mapping) != set(sides) or len(set(mapping.values())) != 3:
        raise ValueError(f'expected three explicit basket assignments: {instruction!r}')
    return [mapping[side] for side in sides]


def pour_pairs(instruction):
    text = normalize(_goal_text(instruction))
    reference = re.compile(r'\b(red|turquoise|violet|white|black|brown)\s+(bottle|bowl)s?\b')
    mentions = list(reference.finditer(text))
    bottles = [match for match in mentions if match[2] == 'bottle']
    bowls = [match for match in mentions if match[2] == 'bowl']
    if not bottles or len(bottles) != len(bowls):
        raise ValueError(f'expected ordered bottle-to-bowl pairs: {instruction!r}')
    if mentions == bottles + bowls and 'respectively' in text:
        links = [(bottles[-1], bowls[0])]
    elif mentions[::2] == bottles and mentions[1::2] == bowls:
        links = zip(bottles, bowls)
    else:
        raise ValueError(f'unclear bottle-to-bowl order: {instruction!r}')
    if any(not re.search(r'\b(?:into|to|over)\b|→|->', text[bottle.end():bowl.start()]) for bottle, bowl in links):
        raise ValueError(f'unclear bottle-to-bowl pair: {instruction!r}')
    source = [match[1] for match in bottles]
    destination = [match[1] for match in bowls]
    if not set(source) <= {'red', 'turquoise', 'violet'} or not set(destination) <= {'white', 'black', 'brown'}:
        raise ValueError(f'invalid bottle or bowl references: {instruction!r}')
    return list(zip(source, destination))


def stack_order(instruction, colours):
    text = normalize(_goal_text(instruction))
    text = re.sub(r'\bthe\s+(top|bottom)\b', r'\1', text)
    reverse = bool(re.search(r'\btop(?:\s+to\s+|\s+)bottom\b', text))
    if reverse and re.search(r'\bbottom(?:\s+to\s+|\s+)top\b', text):
        raise ValueError(f'conflicting stack directions: {instruction!r}')
    colour = r'\b(?:' + '|'.join(map(re.escape, colours)) + r')\b'
    block = rf'{colour}(?:\s+blocks?)?'
    cue = r'\b(?:order(?:\s+of|\s*:)|(?:bottom|top)\s+(?:to\s+)?(?:top|bottom)\s*:?|stack\s+(?:the\s+)?)'
    separator = r'\s*(?:,\s*(?:(?:and|then)\s+)?|\band\s+|\bthen\s+|→|->)\s*'
    sequence = re.search(rf'{cue}\s*({block}(?:{separator}{block})+)', text)
    found = re.findall(colour, sequence[1]) if sequence else []
    if not found or len(found) != len(set(found)):
        raise ValueError(f'ambiguous stack colors: {instruction!r}')
    return found[::-1] if reverse else found


def pickup_target(instruction):
    text = _goal_text(instruction).strip()
    match = re.search(
        r'\b(?:pick up|grasp|lift|raise)\s+(?:the\s+|an?\s+)?(.+?)'
        r'(?=\s+(?:by\s+|more than\s+|at least\s+|\d+(?:\.\d+)?\s*cm\b|and\s+(?:keep|hold|lift|raise)\b)|[,;.!]|$)',
        text, re.I,
    )
    if match is None:
        raise ValueError(f'no pickup target: {instruction!r}')
    target = match[1].strip()
    if re.fullmatch(r'(?:(?:designated|specified|target)\s+)*(?:object|item|it)', target, re.I):
        raise ValueError(f'no concrete pickup target: {instruction!r}')
    height = re.search(r'\b(\d+(?:\.\d+)?)\s*cm\b', text[match.end():], re.I)
    return target, max(0.10, float(height[1]) / 100 if height else 0.10)
