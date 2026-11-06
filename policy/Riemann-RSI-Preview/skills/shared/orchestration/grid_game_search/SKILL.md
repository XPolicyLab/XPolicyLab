---
name: grid_game_search
description: "Computes minimax values for a symbolic 3x3 board and returns empty cell indices with the highest value for player 1; terminal won or full boards return all empty cells."
---

# Grid game search

## Function

Computes minimax values for a symbolic 3x3 board and returns empty cell indices with the highest value for player 1; terminal won or full boards return all empty cells.

## Inputs

- board list/tuple length 9 with int tokens in {-1,0,1}
- player int to cached value: 1 maximizes, -1 minimizes

## Outputs

- run returns list of empty integer cell indices (all tied best moves, or all empty when terminal)
- winner returns winning token or 0
- value returns terminal winner or 0, otherwise minimax score

## Callable interface

Implementation: [implementation.py](implementation.py).

- `winner(board)`
- `value(board, player)`
- `run(board)`

## Constraints

- Rejects non-list/tuple length !=9 or tokens not exact ints in {-1,0,1} with ValueError
- value is lru_cache(maxsize=19683) over symbolic board/player states only
- A detected winning line does not halt the fill-board procedure; remaining empty cells remain legal
- No environment, task, layout, file, geometry, score, or opponent internals are accepted
