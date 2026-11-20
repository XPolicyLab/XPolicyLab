import re


def run(description):
    if (not isinstance(description, str) or not 1 <= len(description) <= 128
            or any(ord(character) < 32 for character in description)):
        raise ValueError('Semantic descriptions use only bounded current public language')
    normalized = re.sub(r'\s+', ' ', description.replace('_', ' ')).strip()
    if not normalized:
        raise ValueError('A current semantic query cannot be empty after normalization')
    alternatives = {
        'basket': ['plastic bin', 'basket', 'container'],
        'car object': ['toy car', 'car', 'car object'],
        'watch object': ['wristwatch', 'watch', 'watch object'],
        'nesting doll': ['nesting doll', 'matryoshka doll', 'wooden doll'],
    }
    if normalized.lower() in alternatives:
        return alternatives[normalized.lower()]
    found = [normalized]
    if normalized.lower().endswith(' object'):
        found.append(normalized[:-7])
    return found
