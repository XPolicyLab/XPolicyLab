import re
_EMB = re.compile('\\b(left|right)\\s+(hand|arm|gripper|end[- ]?effector)\\b', re.I)
_LEAD = re.compile('^\\s*(left|right)\\s+(arm|hand|gripper)\\b\\s*(.*)$', re.I)
_VERB_MAP = {v + s: v for (v, s) in [('pick', 's'), ('place', 's'), ('put', 's'), ('adjust', 's'), ('pour', 's'), ('wipe', 's'), ('stack', 's'), ('straighten', 's'), ('use', 's'), ('push', 'es'), ('close', 's'), ('press', 'es'), ('organize', 's'), ('open', 's'), ('pull', 's'), ('lift', 's'), ('hold', 's'), ('move', 's'), ('grab', 's'), ('take', 's'), ('turn', 's'), ('rotate', 's'), ('fold', 's'), ('insert', 's'), ('remove', 's'), ('hang', 's'), ('drop', 's'), ('slide', 's'), ('tilt', 's'), ('flip', 's'), ('scoop', 's'), ('twist', 's'), ('squeeze', 's'), ('spray', 's'), ('cut', 's'), ('tap', 's'), ('point', 's'), ('give', 's'), ('set', 's'), ('get', 's'), ('bring', 's'), ('wrap', 's'), ('peel', 's'), ('shake', 's'), ('sweep', 's'), ('rub', 's'), ('arrange', 's'), ('position', 's'), ('transfer', 's'), ('align', 's'), ('retrieve', 's'), ('release', 's'), ('grasp', 's'), ('lower', 's'), ('raise', 's'), ('tuck', 's'), ('gather', 's'), ('collect', 's'), ('sort', 's'), ('spread', 's')]}
_COLORS = frozenset({'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'white', 'gray', 'grey', 'brown', 'tan', 'beige', 'gold', 'golden', 'silver', 'bronze', 'cyan', 'magenta', 'maroon', 'navy', 'teal', 'turquoise', 'violet', 'lavender', 'crimson', 'scarlet'})
_MODS = frozenset({'dark', 'light', 'bright', 'pale', 'deep'})
_STOP = frozenset({'and', 'or', 'then', 'it', 'its', 'on', 'onto', 'in', 'into', 'to', 'from', 'at', 'of', 'with', 'near', 'over', 'under', 'by', 'is', 'are', 'was', 'that', 'which', 'up', 'down', 'out', 'off', 'back', 'away', 'aside', 'together', 'a', 'an', 'the', 'one', 'ones'})
_NP_TERM = frozenset({'on', 'in', 'into', 'onto', 'to', 'from', 'at', 'of', 'with', 'and', 'or', 'then', 'near', 'over', 'under', 'by', 'for', 'against', 'around', 'across', 'through', 'between', 'behind', 'beside', 'next', 'off', 'out', 'up', 'down', 'back', 'away', 'it', 'its', 'that', 'which', 'is', 'are', 'was', 'the', 'a', 'an'})
_CALT = '|'.join(sorted(_COLORS, key=len, reverse=True))
_COMPOUND = re.compile('\\b(' + _CALT + ')\\b[\\s,]+(?:and\\s+|or\\s+)?\\b(' + _CALT + ')\\b', re.I)

def _norm(w):
    return re.sub('[^a-z]', '', w.lower())

def hand_free_forms(cap):
    if len(_EMB.findall(cap)) != 1:
        return []
    m = _LEAD.match(cap)
    if not m:
        return []
    noun = m.group(2).lower()
    rest = m.group(3).strip()
    if not rest:
        return []
    forms = ['The ' + noun + ' ' + rest, 'One ' + noun + ' ' + rest]
    low = cap.lower()
    if ' and ' not in low and ', ' not in cap and (' then ' not in low):
        vm = re.match('^(\\w+)(.*)$', rest)
        if vm and _VERB_MAP.get(vm.group(1).lower()):
            b = _VERB_MAP[vm.group(1).lower()]
            forms.append(b[0].upper() + b[1:] + vm.group(2))
    return forms

def _color_noun_chunks(cap):
    toks = cap.split()
    chunks = []
    i = 0
    while i < len(toks):
        if _norm(toks[i]) in _COLORS:
            nxt = _norm(toks[i + 1]) if i + 1 < len(toks) else ''
            if nxt and nxt not in _STOP and (nxt not in _COLORS):
                j = i + 1
                np = []
                while j < len(toks):
                    nt = _norm(toks[j])
                    if not nt or nt in _NP_TERM or nt in _COLORS or (nt in _MODS):
                        break
                    np.append(nt)
                    j += 1
                if np:
                    chunks.append(' '.join(np))
                i = j
                continue
        i += 1
    return chunks

def color_free(cap):
    if _COMPOUND.search(cap):
        return None
    ch = _color_noun_chunks(cap)
    if len(ch) != len(set(ch)):
        return None
    toks = cap.split()
    out = []
    i = 0
    changed = False
    while i < len(toks):
        if _norm(toks[i]) in _COLORS:
            nxt = _norm(toks[i + 1]) if i + 1 < len(toks) else ''
            if nxt and nxt not in _STOP and (nxt not in _COLORS):
                if out and _norm(out[-1]) in _MODS:
                    out.pop()
                changed = True
                i += 1
                continue
        out.append(toks[i])
        i += 1
    res = re.sub('\\s+', ' ', ' '.join(out)).strip()
    return res if changed else None

def drop_cues(caption, drop_hand, drop_color, rng):
    if not caption or not isinstance(caption, str):
        return caption
    c = caption
    if drop_hand:
        forms = hand_free_forms(c)
        if forms:
            c = forms[int(rng.integers(len(forms)))]
    if drop_color:
        cf = color_free(c)
        if cf is not None:
            c = cf
    return c
