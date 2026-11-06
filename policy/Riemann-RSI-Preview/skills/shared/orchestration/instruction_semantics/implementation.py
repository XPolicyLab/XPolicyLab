"""Bounded public action/object semantics, never task identifiers or scene facts."""
from dataclasses import dataclass
from functools import lru_cache, wraps
import re
import unicodedata

from harness.execution import ExecutionError


ACTION_WORDS = {
    'swap': ('swap', 'exchange', 'interchange', 'switch', 'trade'),
    'stack': ('stack', 'pile', 'nest'),
    'pour': ('pour', 'decant'),
    'push': ('push', 'slide', 'nudge', 'shove'),
    'move': ('move', 'relocate', 'transfer'),
    'align': ('align', 'alignment', 'straighten'),
    'press': ('press', 'depress', 'tap'),
    'confirm': ('confirm', 'confirmation', 'acknowledge'),
    'insert': ('insert', 'plug', 'slot'),
    'place': ('place', 'put', 'position', 'set', 'route'),
    'store': ('store', 'stow', 'pack', 'organize', 'organise'),
    'cover': ('cover', 'uncover', 'hide', 'reveal'),
    'play': ('play', 'strike', 'hit', 'sound'),
    'fold': ('fold', 'crease'),
    'arrange': ('arrange', 'form', 'make', 'build', 'order', 'sort'),
    'complete': ('complete', 'solve', 'finish'),
    'hold': ('hold', 'support'), 'close': ('close', 'shut'),
    'tighten': ('tighten', 'fasten', 'screw'), 'hang': ('hang', 'hook', 'suspend'),
    'remember': ('remember', 'memorize', 'memorise'), 'observe': ('observe', 'watch'),
    'turn': ('turn', 'twist', 'rotate'), 'lift': ('lift', 'raise'),
    'pick': ('pick', 'pickup', 'grasp', 'grab', 'take'), 'match': ('match', 'matching'),
}
OBJECT_WORDS = {
    'block': ('block', 'cube'), 'bowl': ('bowl',),
    'bottle': ('bottle', 'flask'), 'liquid': ('liquid', 'water', 'fluid'),
    'cup': ('cup', 'glass', 'mug', 'beaker'), 'ball': ('ball', 'sphere'),
    'vase': ('vase',), 'mat': ('mat', 'pad'), 'button': ('button',),
    'card': ('card',), 'number': ('number', 'numeral', 'digit', 'integer'),
    'tube': ('tube',), 'rack': ('rack', 'holder'), 'doll': ('doll', 'matryoshka'),
    'garment': ('garment', 'clothes', 'cloth', 'shirt'),
    'toolbox': ('toolbox',), 'xylophone': ('xylophone',), 'mallet': ('mallet',),
    'board': ('board', 'checkerboard'), 'bread': ('bread',), 'toaster': ('toaster',),
    'object': ('object', 'item', 'thing'), 'basket': ('basket',), 'box': ('box',),
    'coin': ('coin',), 'bank': ('bank', 'piggybank'), 'key': ('key',), 'lock': ('lock', 'keyhole'),
    'screw': ('screw', 'bolt'), 'nut': ('nut',), 'egg': ('egg',), 'lid': ('lid',),
    'pen': ('pen',), 'peg': ('peg', 'post'), 'ring': ('ring',), 'toy': ('toy',),
    'conveyor': ('conveyor',), 'charger': ('charger', 'adapter'), 'strip': ('strip', 'socket', 'outlet'),
    'clock': ('clock',), 'drawer': ('drawer',), 'figurine': ('figurine', 'statuette'),
    'mouse': ('mouse',), 'keyboard': ('keyboard',), 'frame': ('frame',),
}
TERM_WORDS = {'empty': ('empty', 'vacant', 'unoccupied'),
              'missing': ('missing', 'absent'), 'largest': ('largest', 'biggest', 'maximum', 'highest'),
              'smallest': ('smallest', 'smallest-sized', 'minimum'),
              'each': ('each', 'every'), 'texture': ('texture', 'textured', 'pattern', 'patterned'),
              'colour': ('colour', 'color'), 'category': ('category', 'type', 'kind', 'class'),
              'other': ('other', 'opposite', 'different'), 'back': ('back', 'return', 'restore'),
              'front': ('front', 'forward'), 'correct': ('correct', 'suitable', 'appropriate'),
              'image': ('image', 'picture', 'photo'),
              'same': ('same', 'identical', 'identically'), 'again': ('again', 'reappear')}
NEGATIONS = frozenset(('not', 'never', 'avoid', 'without', "don't", "dont"))
ALIASES = {word: canonical for groups in (ACTION_WORDS, OBJECT_WORDS, TERM_WORDS)
           for canonical, words in groups.items() for word in words}


def canonical_word(word):
    if word.endswith("'s"):
        word = word[:-2]
    if word in ALIASES:
        return ALIASES[word]
    for suffix in ('ing', 'ed', 'es', 's'):
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            for candidate in (stem, stem + 'e', stem[:-1] if len(stem) > 2 and stem[-1:] == stem[-2:-1] else stem):
                if candidate in ALIASES:
                    return ALIASES[candidate]
    return word


@dataclass(frozen=True)
class InstructionSemantics:
    terms: frozenset
    actions: frozenset
    objects: frozenset
    relations: frozenset


def parse(instruction):
    if instruction is None:
        instruction = ''
    if not isinstance(instruction, str) or len(instruction) > 32768:
        raise ExecutionError('Instruction semantics accepts only bounded original public text')
    return cached_parse(instruction)


@lru_cache(maxsize=8)
def cached_parse(instruction):
    text = unicodedata.normalize('NFKC', instruction).casefold().replace('’', "'")
    tokens = tuple(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.;!?]", text))
    words = tuple(canonical_word(token) for token in tokens)
    terms = frozenset(tokens) | frozenset(words)
    negated = set()
    for index, token in enumerate(tokens):
        if token not in NEGATIONS:
            continue
        for candidate in words[index + 1:index + 13]:
            if candidate in ('.', ';', '!', '?', 'but', 'instead'):
                break
            if candidate in ACTION_WORDS or candidate == 'empty':
                negated.add(candidate)
    actions = set(words) & set(ACTION_WORDS) - negated
    if ('line' in terms and 'up' in terms) or {'in', 'line'} <= terms:
        if 'align' not in negated:
            actions.add('align')
    objects = set(words) & set(OBJECT_WORDS)
    if 't' in terms and not {'shirt', 'shirted'}.intersection(terms):
        objects.add('t_shape')
    if 'tool' in terms and 'box' in terms:
        objects.add('toolbox')
    if {'tic', 'tac', 'toe'} <= terms:
        objects.add('game_board')
    relations = set()
    if 'card' in objects and 'number' in objects and ('times' in terms or 'repeat' in terms or 'count' in terms):
        relations.add('card_repeat_count')
    if ('each' in terms and 'move' in actions and ('after' in terms or 'following' in terms)
            and ('press' in actions or 'confirm' in actions)):
        relations.add('confirm_after_each_move')
    if ('empty' in terms and 'empty' not in negated and 'bottle' in objects and 'cup' in objects
            and ('into' in terms or 'to' in terms)):
        actions.add('pour')
    if ('smallest' in terms and 'largest' in terms and 'to' in terms
            and words.index('smallest') < words.index('largest')) or 'ascending' in terms:
        relations.add('ascending_size')
    if 'descending' in terms:
        relations.add('descending_size')
    if 'colour' in terms and ('same' in terms or 'match' in actions):
        relations.add('matching_colour')
    if 'other' in terms and ('hand' in terms or 'arm' in terms):
        relations.add('different_arm')
    if ('hand' in terms and 'over' in terms) or 'handover' in terms or 'pass' in terms:
        relations.add('handover')
    if 'same' in terms and ('order' in terms or 'sequence' in terms):
        relations.add('same_order')
    return InstructionSemantics(terms, frozenset(actions), frozenset(objects), frozenset(relations))


def matches(instruction, *, actions=(), all_actions=(), objects=(), terms=(), relations=(), any_objects=(), any_terms=(), term_groups=(), exclude_objects=()):
    value = parse(instruction)
    return (bool(set(actions) & value.actions) if actions else True) and (
        set(all_actions) <= value.actions and set(objects) <= value.objects and set(terms) <= value.terms
        and set(relations) <= value.relations
        and (bool(set(any_objects) & value.objects) if any_objects else True)
        and (bool(set(any_terms) & value.terms) if any_terms else True)
        and all(set(group) & value.terms for group in term_groups)
        and not set(exclude_objects) & value.objects)


def prior_terms(instruction):
    value = parse(instruction)
    words = set(value.terms) | set(value.objects) | set(value.actions)
    words |= {word[:-1] for word in tuple(words) if word.endswith('s')}
    return words


def resolver_contract(instruction_predicate):
    """Decorate a resolver with public applicability, preserving all original result fields.

    Input is a callable on public instruction text; output is a resolver decorator. The predicate
    must not read perception, task identifiers, priors, execution state or official outcomes.
    An already returned relation also claims an existing continuation/cleanup; no relation is created.
    """
    def decorate(resolver):
        """Accept an API/history resolver callable; return the same callable with a boolean result field."""
        @wraps(resolver)
        def resolve(api, memory, maximum_uncertain_attempts=2):
            """Accept public robot API, event-list memory and attempt bound; return an annotated result dict."""
            result = resolver(api, memory, maximum_uncertain_attempts)
            return dict(result, applicable=bool(instruction_predicate(api.instruction()) or result['relation'] is not None))
        return resolve
    return decorate
