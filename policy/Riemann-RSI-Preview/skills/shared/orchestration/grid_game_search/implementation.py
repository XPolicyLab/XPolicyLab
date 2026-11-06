from functools import lru_cache


LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def winner(board):
    for first, second, third in LINES:
        if board[first] and board[first] == board[second] == board[third]:
            return board[first]
    return 0


@lru_cache(maxsize=19683)
def value(board, player):
    winning = winner(board)
    if winning:
        return winning
    options = [identity for identity, token in enumerate(board) if not token]
    if not options:
        return 0
    evaluations = []
    for identity in options:
        updated = list(board)
        updated[identity] = player
        evaluations.append(value(tuple(updated), -player))
    return (max if player == 1 else min)(evaluations)


def run(board):
    if (not isinstance(board, (list, tuple)) or len(board) != 9
            or any(type(token) is not int or token not in (-1, 0, 1) for token in board)):
        raise ValueError('Grid search accepts only a bounded publicly reconstructed symbolic board')
    empty = [identity for identity, token in enumerate(board) if not token]
    if winner(board) or not empty:
        return empty
    scores = {}
    for identity in empty:
        updated = list(board)
        updated[identity] = 1
        scores[identity] = value(tuple(updated), -1)
    highest = max(scores.values())
    return [identity for identity in empty if scores[identity] == highest]
