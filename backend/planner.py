"""Fast, dependency-free expectimax play for 2048.

The learner is intentionally small and is useful for experimentation, but a
dashboard should also have a dependable evaluation policy.  This module keeps
that policy separate from training: rows are represented as four-bit tile
exponents and cached row transitions make a two-ply expectimax search cheap
enough for CPU-only runs.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from threading import RLock
from typing import Sequence

from .game import Game2048


_ROW_LEFT = [0] * 65_536
_ROW_RIGHT = [0] * 65_536
_ROW_EMPTY = [0] * 65_536
_ROW_SMOOTH = [0.0] * 65_536
_ROW_MONOTONIC = [0.0] * 65_536

# A serpentine positional pattern anchored at the upper-left corner.  It
# rewards keeping the largest tile in a stable corner while leaving room for
# the lower rows to feed it.
_POSITION_WEIGHTS = (16, 15, 14, 13, 8, 9, 10, 11, 7, 6, 5, 4, 0, 1, 2, 3)
_NATIVE_LOCK = RLock()
_NATIVE_LIBRARY: ctypes.CDLL | None = None
_NATIVE_FAILED = False


def _reverse_row(value: int) -> int:
    return (
        ((value & 0x000F) << 12)
        | ((value & 0x00F0) << 4)
        | ((value & 0x0F00) >> 4)
        | ((value & 0xF000) >> 12)
    )


def _build_tables() -> None:
    # Left moves can be built independently.  Right moves are the mirrored
    # left transition, but must be built after every left row is available.
    for row in range(65_536):
        values = [(row >> (4 * index)) & 0xF for index in range(4)]
        packed = [value for value in values if value]
        merged: list[int] = []
        index = 0
        while index < len(packed):
            if index + 1 < len(packed) and packed[index] == packed[index + 1]:
                merged.append(packed[index] + 1)
                index += 2
            else:
                merged.append(packed[index])
                index += 1
        _ROW_LEFT[row] = sum(value << (4 * index) for index, value in enumerate(merged))
        _ROW_EMPTY[row] = values.count(0)
        _ROW_SMOOTH[row] = -sum(
            abs(values[index] - values[index + 1])
            for index in range(3)
            if values[index] and values[index + 1]
        )
        increasing = sum(max(0, values[index + 1] - values[index]) for index in range(3))
        decreasing = sum(max(0, values[index] - values[index + 1]) for index in range(3))
        _ROW_MONOTONIC[row] = float(max(increasing, decreasing))
    for row in range(65_536):
        _ROW_RIGHT[row] = _reverse_row(_ROW_LEFT[_reverse_row(row)])


_build_tables()


def _move(board: int, action: int) -> int:
    """Move a packed exponent board without allocating Python board lists."""

    result = 0
    if action in (1, 3):  # right/left: rows
        table = _ROW_RIGHT if action == 1 else _ROW_LEFT
        for row in range(4):
            value = table[(board >> (16 * row)) & 0xFFFF]
            result |= value << (16 * row)
        return result

    # up/down: read each column as a row, then write it back transposed.
    table = _ROW_LEFT if action == 0 else _ROW_RIGHT
    for column in range(4):
        value = 0
        for row in range(4):
            value |= ((board >> (4 * (4 * row + column))) & 0xF) << (4 * row)
        moved = table[value]
        for row in range(4):
            result |= ((moved >> (4 * row)) & 0xF) << (4 * (4 * row + column))
    return result


def _native_library() -> ctypes.CDLL | None:
    """Build/load the optional C++ accelerator on first deep-search request."""

    global _NATIVE_LIBRARY, _NATIVE_FAILED
    if _NATIVE_LIBRARY is not None or _NATIVE_FAILED:
        return _NATIVE_LIBRARY
    if os.environ.get("LATENT_2048_DISABLE_NATIVE"):
        _NATIVE_FAILED = True
        return None
    source = Path(__file__).with_name("native_planner.cpp")
    target = Path(tempfile.gettempdir()) / "latent_2048_native_planner.so"
    try:
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(
                ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(target)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        library = ctypes.CDLL(str(target))
        library.choose_action.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        library.choose_action.restype = ctypes.c_int
        library.clear_planner_cache.argtypes = []
        library.clear_planner_cache.restype = None
        _NATIVE_LIBRARY = library
    except (OSError, subprocess.SubprocessError):
        _NATIVE_FAILED = True
    return _NATIVE_LIBRARY


def _heuristic(board: int) -> float:
    empty = 0
    smoothness = 0.0
    monotonicity = 0.0
    positional = sum(
        ((board >> (4 * index)) & 0xF) * weight
        for index, weight in enumerate(_POSITION_WEIGHTS)
    )
    maximum = 0
    for row in range(4):
        value = (board >> (16 * row)) & 0xFFFF
        empty += _ROW_EMPTY[value]
        smoothness += _ROW_SMOOTH[value]
        monotonicity += _ROW_MONOTONIC[value]
        maximum = max(maximum, *(value >> (4 * index) & 0xF for index in range(4)))
    for column in range(4):
        value = 0
        for row in range(4):
            value |= ((board >> (4 * (4 * row + column))) & 0xF) << (4 * row)
        smoothness += _ROW_SMOOTH[value]
        monotonicity += _ROW_MONOTONIC[value]
    # Empty cells are deliberately dominant: surviving long enough to form
    # the target is more important than a locally attractive merge.
    corner_bonus = 500.0 if (board & 0xF) == maximum else 0.0
    return empty * 300.0 + positional * 4.0 + monotonicity * 20.0 + smoothness * 30.0 + corner_bonus


@lru_cache(maxsize=3_000_000)
def _expectimax(board: int, depth: int) -> float:
    if depth <= 0:
        return _heuristic(board)
    best = -math.inf
    for action in range(4):
        moved = _move(board, action)
        if moved == board:
            continue
        empty = [index for index in range(16) if not ((moved >> (4 * index)) & 0xF)]
        if not empty:
            value = _heuristic(moved)
        else:
            value = sum(
                (
                    0.9 * _expectimax(moved | (1 << (4 * index)), depth - 1)
                    + 0.1 * _expectimax(moved | (2 << (4 * index)), depth - 1)
                )
                / len(empty)
                for index in empty
            )
        best = max(best, value)
    return _heuristic(board) if best == -math.inf else best


def packed_board(board: Sequence[int]) -> int:
    """Convert a flat board of tile values to packed base-2 exponents."""

    if len(board) != 16:
        raise ValueError("a packed planner board must contain 16 cells")
    packed = 0
    for index, value in enumerate(board):
        if value:
            packed |= int(math.log2(int(value))) << (4 * index)
    return packed


def _exponents(board: Sequence[int]) -> list[int]:
    if len(board) != 16:
        raise ValueError("a planner board must contain 16 cells")
    return [int(math.log2(int(value))) if value else 0 for value in board]


def choose_action(board: Sequence[int], *, depth: int = 2) -> int | None:
    """Return the strongest legal action for a flat 2048 board.

    ``depth=2`` means one action/chance layer followed by a second action
    layer.  It is a useful balance for a live CPU dashboard; callers that can
    spend more time may request depth 3.
    """

    exponents = _exponents(board)
    if depth >= 3:
        library = _native_library()
        if library is not None:
            cells = (ctypes.c_uint8 * 16)(*exponents)
            with _NATIVE_LOCK:
                action = int(library.choose_action(cells, int(depth)))
            return action if action >= 0 else None
    packed = sum(value << (4 * index) for index, value in enumerate(exponents))
    best_value = -math.inf
    best_action: int | None = None
    for action in range(4):
        moved = _move(packed, action)
        if moved == packed:
            continue
        empty = [index for index in range(16) if not ((moved >> (4 * index)) & 0xF)]
        if not empty:
            value = _heuristic(moved)
        else:
            value = sum(
                (
                    0.9 * _expectimax(moved | (1 << (4 * index)), depth - 1)
                    + 0.1 * _expectimax(moved | (2 << (4 * index)), depth - 1)
                )
                / len(empty)
                for index in empty
            )
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def play_expectimax_game(
    seed: int = 2059,
    *,
    max_steps: int = 5_000,
    target_tile: int = 4_096,
    depth: int = 3,
) -> dict:
    """Play a complete evaluation game with the expectimax policy."""

    clear_cache()
    game = Game2048(random.Random(seed))
    initial_board = game.board.copy()
    moves: list[dict] = []
    while not game.is_game_over() and game.steps < max_steps and game.max_tile < target_tile:
        action = choose_action(game.board, depth=depth)
        if action is None:
            break
        before = game.board.copy()
        result = game.step(action)
        moves.append({**result.to_dict(), "board_before": before})
    result = {
        **game.as_dict(),
        "initial_board": initial_board,
        "moves": moves,
        "policy": "expectimax",
        "target_tile": target_tile,
        "target_reached": game.max_tile >= target_tile,
    }
    clear_cache()
    return result


def clear_cache() -> None:
    """Release transposition-table memory between long evaluation runs."""

    _expectimax.cache_clear()
    if _NATIVE_LIBRARY is not None:
        with _NATIVE_LOCK:
            _NATIVE_LIBRARY.clear_planner_cache()


__all__ = ["choose_action", "clear_cache", "packed_board", "play_expectimax_game"]
