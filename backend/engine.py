"""Deterministic, serializable 2048 environment.

The environment keeps the rules independent from rendering or the learning
algorithm.  A seeded :class:`GameState` is therefore suitable for replays,
tests, and self-play evaluation.  Boards are immutable tuples at the rule
boundary, which makes accidental mutation by an agent difficult.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import pickle
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Mapping, Sequence

SIZE = 4
TARGET = 2048
Board = tuple[tuple[int, ...], ...]


class Direction(IntEnum):
    """The action order used by the API and by trained models."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @classmethod
    def parse(cls, value: "Direction | str | int") -> "Direction":
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value)
        text = str(value).strip().lower()
        aliases = {
            "up": cls.UP,
            "arrowup": cls.UP,
            "north": cls.UP,
            "right": cls.RIGHT,
            "arrowright": cls.RIGHT,
            "east": cls.RIGHT,
            "down": cls.DOWN,
            "arrowdown": cls.DOWN,
            "south": cls.DOWN,
            "left": cls.LEFT,
            "arrowleft": cls.LEFT,
            "west": cls.LEFT,
        }
        if text not in aliases:
            raise ValueError(f"unknown direction: {value!r}")
        return aliases[text]

    @property
    def name_lower(self) -> str:
        return self.name.lower()


def _normalise_board(board: Sequence[Sequence[int]]) -> Board:
    if len(board) != SIZE or any(len(row) != SIZE for row in board):
        raise ValueError(f"a board must be {SIZE}x{SIZE}")
    result: list[tuple[int, ...]] = []
    for row in board:
        normal_row: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("board values must be non-negative integers")
            if value and value & (value - 1):
                raise ValueError("board values must be powers of two")
            normal_row.append(value)
        result.append(tuple(normal_row))
    return tuple(result)


def empty_board() -> Board:
    return ((0, 0, 0, 0),) * SIZE


def _slide_line(line: Sequence[int]) -> tuple[tuple[int, ...], int, int]:
    """Slide one line towards its first element.

    Returns ``(line, score_delta, merge_count)``.  Keeping this function
    pure makes the most error-prone part of 2048 easy to test.
    """

    compact = [value for value in line if value]
    merged: list[int] = []
    score = 0
    merges = 0
    index = 0
    while index < len(compact):
        if index + 1 < len(compact) and compact[index] == compact[index + 1]:
            value = compact[index] * 2
            merged.append(value)
            score += value
            merges += 1
            index += 2
        else:
            merged.append(compact[index])
            index += 1
    merged.extend([0] * (SIZE - len(merged)))
    return tuple(merged), score, merges


@dataclass(frozen=True)
class MoveResult:
    board: Board
    changed: bool
    score_delta: int
    merges: int
    spawned: tuple[int, int, int] | None = None
    reward: float = 0.0
    done: bool = False
    won: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "board": [list(row) for row in self.board],
            "changed": self.changed,
            "score_delta": self.score_delta,
            "merges": self.merges,
            "spawned": list(self.spawned) if self.spawned is not None else None,
            "reward": self.reward,
            "done": self.done,
            "won": self.won,
        }


def move_board(board: Sequence[Sequence[int]], direction: Direction | str | int) -> MoveResult:
    """Apply a move without spawning a tile.

    This pure primitive is useful when an agent wants to simulate moves.  The
    stateful :meth:`GameState.step` calls it and then performs the random tile
    spawn only when the move changed the board.
    """

    board = _normalise_board(board)
    direction = Direction.parse(direction)
    rows = [list(row) for row in board]
    output = [[0] * SIZE for _ in range(SIZE)]
    score = 0
    merges = 0

    if direction is Direction.LEFT:
        for row in range(SIZE):
            output[row], row_score, row_merges = _slide_line(rows[row])
            score += row_score
            merges += row_merges
    elif direction is Direction.RIGHT:
        for row in range(SIZE):
            slid, row_score, row_merges = _slide_line(reversed(rows[row]))
            output[row] = list(reversed(slid))
            score += row_score
            merges += row_merges
    elif direction is Direction.UP:
        for column in range(SIZE):
            slid, column_score, column_merges = _slide_line(rows[row][column] for row in range(SIZE))
            for row, value in enumerate(slid):
                output[row][column] = value
            score += column_score
            merges += column_merges
    else:  # DOWN
        for column in range(SIZE):
            slid, column_score, column_merges = _slide_line(rows[row][column] for row in reversed(range(SIZE)))
            for row, value in zip(reversed(range(SIZE)), slid):
                output[row][column] = value
            score += column_score
            merges += column_merges

    result = tuple(tuple(row) for row in output)
    return MoveResult(result, result != board, score, merges)


def legal_moves(board: Sequence[Sequence[int]]) -> tuple[Direction, ...]:
    """Return actions that would change the board in canonical action order."""

    board = _normalise_board(board)
    return tuple(direction for direction in Direction if move_board(board, direction).changed)


def max_tile(board: Sequence[Sequence[int]]) -> int:
    return max(max(row) for row in board)


def empty_count(board: Sequence[Sequence[int]]) -> int:
    return sum(value == 0 for row in board for value in row)


def _spawn(board: Board, rng: random.Random) -> tuple[Board, tuple[int, int, int] | None]:
    empty = [(row, column) for row in range(SIZE) for column in range(SIZE) if board[row][column] == 0]
    if not empty:
        return board, None
    row, column = rng.choice(empty)
    value = 4 if rng.random() < 0.1 else 2
    mutable = [list(line) for line in board]
    mutable[row][column] = value
    return tuple(tuple(line) for line in mutable), (row, column, value)


def _rng_to_string(rng: random.Random) -> str:
    return base64.b85encode(pickle.dumps(rng.getstate(), protocol=4)).decode("ascii")


def _rng_from_string(encoded: str) -> random.Random:
    rng = random.Random()
    state = pickle.loads(base64.b85decode(encoded.encode("ascii")))
    rng.setstate(state)
    return rng


@dataclass(frozen=True)
class Turn:
    """One attempted action, including no-op actions for faithful replays."""

    index: int
    action: str
    board_before: Board
    board_after: Board
    changed: bool
    score_delta: int
    reward: float
    spawned: tuple[int, int, int] | None
    done: bool
    won: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "board_before": [list(row) for row in self.board_before],
            "board_after": [list(row) for row in self.board_after],
            "changed": self.changed,
            "score_delta": self.score_delta,
            "reward": self.reward,
            "spawned": list(self.spawned) if self.spawned is not None else None,
            "done": self.done,
            "won": self.won,
        }


def reward_for_move(
    before: Board,
    after: Board,
    *,
    score_delta: int,
    changed: bool,
    done: bool,
) -> float:
    """Reward shaped for survival first, with score and milestones retained.

    The raw 2048 score is still exposed separately.  The small survival term
    gives a learner a useful signal on quiet moves, while milestone shaping
    nudges it toward larger tiles.  Invalid moves are penalised lightly and a
    terminal board receives a clear penalty.
    """

    if not changed:
        return -0.25
    before_max = max_tile(before)
    after_max = max_tile(after)
    milestone = max(0.0, math.log2(after_max / before_max)) if before_max else 0.0
    reward = 1.0 + 0.01 * score_delta + 0.35 * milestone
    if done:
        reward -= 5.0
    return reward


@dataclass
class GameState:
    """Mutable episode state with a deterministic random stream."""

    board: Board
    score: int = 0
    moves: int = 0
    seed: int | None = None
    won: bool = False
    done: bool = False
    history: list[Turn] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    @classmethod
    def new(cls, seed: int | None = None) -> "GameState":
        rng = random.Random(seed)
        board, _ = _spawn(empty_board(), rng)
        board, _ = _spawn(board, rng)
        return cls(board=board, seed=seed, _rng=rng)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GameState":
        state = cls(
            board=_normalise_board(payload["board"]),
            score=int(payload.get("score", 0)),
            moves=int(payload.get("moves", 0)),
            seed=payload.get("seed"),
            won=bool(payload.get("won", False)),
            done=bool(payload.get("done", False)),
        )
        encoded_rng = payload.get("rng_state")
        if encoded_rng:
            state._rng = _rng_from_string(str(encoded_rng))
        state.history = [
            Turn(
                index=int(turn["index"]),
                action=str(turn["action"]),
                board_before=_normalise_board(turn["board_before"]),
                board_after=_normalise_board(turn["board_after"]),
                changed=bool(turn["changed"]),
                score_delta=int(turn["score_delta"]),
                reward=float(turn["reward"]),
                spawned=tuple(turn["spawned"]) if turn.get("spawned") is not None else None,
                done=bool(turn["done"]),
                won=bool(turn["won"]),
            )
            for turn in payload.get("history", [])
        ]
        return state

    def clone(self, *, include_history: bool = False) -> "GameState":
        clone = GameState.from_dict(self.to_dict(include_history=include_history))
        return clone

    @property
    def highest_tile(self) -> int:
        return max_tile(self.board)

    @property
    def legal_moves(self) -> tuple[Direction, ...]:
        return legal_moves(self.board)

    def step(self, action: Direction | str | int) -> MoveResult:
        direction = Direction.parse(action)
        before = self.board
        pure = move_board(before, direction)
        spawned = None
        after = pure.board
        if pure.changed:
            after, spawned = _spawn(after, self._rng)
            self.moves += 1
            self.score += pure.score_delta
        self.board = after
        self.won = self.won or max_tile(after) >= TARGET
        self.done = not legal_moves(after)
        reward = reward_for_move(
            before,
            after,
            score_delta=pure.score_delta,
            changed=pure.changed,
            done=self.done,
        )
        turn = Turn(
            index=len(self.history),
            action=direction.name_lower,
            board_before=before,
            board_after=after,
            changed=pure.changed,
            score_delta=pure.score_delta,
            reward=reward,
            spawned=spawned,
            done=self.done,
            won=self.won,
        )
        self.history.append(turn)
        return MoveResult(
            board=after,
            changed=pure.changed,
            score_delta=pure.score_delta,
            merges=pure.merges,
            spawned=spawned,
            reward=reward,
            done=self.done,
            won=self.won,
        )

    def reset(self, seed: int | None = None) -> None:
        fresh = self.new(seed)
        self.board = fresh.board
        self.score = fresh.score
        self.moves = fresh.moves
        self.seed = fresh.seed
        self.won = False
        self.done = False
        self.history.clear()
        self._rng = fresh._rng

    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        return {
            "board": [list(row) for row in self.board],
            "score": self.score,
            "moves": self.moves,
            "seed": self.seed,
            "highest_tile": self.highest_tile,
            "legal_moves": [direction.name_lower for direction in self.legal_moves],
            "won": self.won,
            "done": self.done,
            "rng_state": _rng_to_string(self._rng),
            "history": [turn.to_dict() for turn in self.history] if include_history else [],
        }

    def to_json(self, *, include_history: bool = True) -> str:
        return json.dumps(self.to_dict(include_history=include_history), separators=(",", ":"))


def new_game(seed: int | None = None) -> GameState:
    return GameState.new(seed)


__all__ = [
    "Board",
    "Direction",
    "GameState",
    "MoveResult",
    "SIZE",
    "TARGET",
    "Turn",
    "empty_board",
    "empty_count",
    "legal_moves",
    "max_tile",
    "move_board",
    "new_game",
    "reward_for_move",
]
