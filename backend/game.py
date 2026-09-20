"""Small, dependency-light 2048 environment used by the trainer and API.

The environment intentionally has no rendering or framework dependencies.  It
is deterministic when constructed with a seeded ``random.Random`` instance,
which makes checkpoints and local experiments reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import numpy as np


ACTIONS = ("up", "right", "down", "left")
ACTION_TO_ID = {name: i for i, name in enumerate(ACTIONS)}


@dataclass(frozen=True)
class StepResult:
    action: int
    action_name: str
    changed: bool
    score_delta: int
    reward: float
    done: bool
    board: tuple[int, ...]
    score: int
    max_tile: int
    empty_cells: int

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "changed": self.changed,
            "score_delta": self.score_delta,
            "reward": round(float(self.reward), 6),
            "done": self.done,
            "board": list(self.board),
            "score": self.score,
            "max_tile": self.max_tile,
            "empty_cells": self.empty_cells,
        }


class Game2048:
    """The standard 4x4 2048 rules, with a compact observation helper."""

    size = 4

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.board: list[int] = [0] * 16
        self.score = 0
        self.steps = 0
        self.reset()

    def reset(self, board: Sequence[int] | None = None, score: int = 0) -> np.ndarray:
        if board is None:
            self.board = [0] * 16
            self._spawn()
            self._spawn()
        else:
            if len(board) != 16:
                raise ValueError("A 2048 board must contain 16 cells")
            if any(v < 0 or (v and (v & (v - 1))) for v in board):
                raise ValueError("Board values must be powers of two or zero")
            self.board = [int(v) for v in board]
        self.score = int(score)
        self.steps = 0
        return self.observation()

    def clone(self) -> "Game2048":
        other = Game2048(self.rng)
        other.board = self.board.copy()
        other.score = self.score
        other.steps = self.steps
        return other

    @property
    def max_tile(self) -> int:
        return max(self.board, default=0)

    @property
    def empty_cells(self) -> int:
        return self.board.count(0)

    def observation(self) -> np.ndarray:
        # A tile of 2 maps to 1/16, while zero remains exactly zero.  Clipping
        # keeps giant late-game tiles in the same numerically useful range.
        return np.asarray(
            [min(1.0, math.log2(v) / 16.0) if v else 0.0 for v in self.board],
            dtype=np.float32,
        )

    def legal_actions(self) -> list[int]:
        return [a for a in range(4) if self.peek(a)[0]]

    def is_game_over(self) -> bool:
        return not self.legal_actions()

    def step(self, action: int | str) -> StepResult:
        action_id = self._action_id(action)
        changed, next_board, score_delta = self._transform(action_id)
        previous_empty = self.empty_cells
        if changed:
            self.board = next_board
            self.score += score_delta
            self._spawn()
        self.steps += 1
        done = self.is_game_over()
        # Valid moves carry a small survival reward.  Score gain is deliberately
        # scaled so the policy cannot maximize one flashy merge while dying.
        if changed:
            reward = 0.10 + score_delta / 64.0 + max(0, self.empty_cells - previous_empty) * 0.01
        else:
            reward = -0.12
        if done:
            reward -= 1.0
        return StepResult(
            action=action_id,
            action_name=ACTIONS[action_id],
            changed=changed,
            score_delta=score_delta,
            reward=float(reward),
            done=done,
            board=tuple(self.board),
            score=self.score,
            max_tile=self.max_tile,
            empty_cells=self.empty_cells,
        )

    def peek(self, action: int | str) -> tuple[bool, list[int], int]:
        """Return a move result without spawning a random tile or mutating state."""
        return self._transform(self._action_id(action))

    def heuristic_action(self) -> int | None:
        """A strong, deterministic fallback used by the dashboard demo.

        This is not used for learning.  It provides a useful game immediately
        after startup while the recurrent policy is still warming up.
        """
        best: tuple[float, int] | None = None
        for action in self.legal_actions():
            changed, candidate, gain = self.peek(action)
            if not changed:
                continue
            empties = candidate.count(0)
            smoothness = self._smoothness(candidate)
            maximum = max(candidate)
            corner_tiles = (candidate[0], candidate[3], candidate[12], candidate[15])
            corner_bonus = math.log2(maximum) if maximum and maximum in corner_tiles else 0.0
            value = empties * 2.7 + gain / 48.0 + smoothness * 0.35 + corner_bonus * 0.65
            # Keep the largest tile in a corner when ties are close.
            if best is None or value > best[0]:
                best = (value, action)
        return best[1] if best else None

    def as_dict(self) -> dict:
        return {
            "board": self.board.copy(),
            "score": self.score,
            "steps": self.steps,
            "max_tile": self.max_tile,
            "empty_cells": self.empty_cells,
            "game_over": self.is_game_over(),
        }

    def _spawn(self) -> None:
        empty = [i for i, value in enumerate(self.board) if value == 0]
        if not empty:
            return
        idx = self.rng.choice(empty)
        self.board[idx] = 4 if self.rng.random() < 0.1 else 2

    @staticmethod
    def _merge_line(line: Sequence[int]) -> tuple[list[int], int]:
        packed = [v for v in line if v]
        merged: list[int] = []
        score = 0
        i = 0
        while i < len(packed):
            if i + 1 < len(packed) and packed[i] == packed[i + 1]:
                value = packed[i] * 2
                merged.append(value)
                score += value
                i += 2
            else:
                merged.append(packed[i])
                i += 1
        return merged + [0] * (4 - len(merged)), score

    def _transform(self, action: int) -> tuple[bool, list[int], int]:
        if action not in range(4):
            raise ValueError(f"action must be one of 0..3, got {action!r}")
        candidate = [0] * 16
        score = 0
        for index in range(4):
            if action in (1, 2):  # right/down: read the line in reverse order
                indices = self._line_indices(action, index, reverse=True)
            else:
                indices = self._line_indices(action, index, reverse=False)
            line, gained = self._merge_line([self.board[i] for i in indices])
            score += gained
            for board_index, value in zip(indices, line):
                candidate[board_index] = value
        return candidate != self.board, candidate, score

    @staticmethod
    def _line_indices(action: int, index: int, reverse: bool = False) -> list[int]:
        if action in (0, 2):  # columns
            result = [index + row * 4 for row in range(4)]
        else:  # rows
            result = [index * 4 + col for col in range(4)]
        return result[::-1] if reverse else result

    @staticmethod
    def _action_id(action: int | str) -> int:
        if isinstance(action, str):
            normalized = action.strip().lower()
            if normalized not in ACTION_TO_ID:
                raise ValueError(f"unknown action {action!r}; expected {ACTIONS}")
            return ACTION_TO_ID[normalized]
        return int(action)

    @staticmethod
    def _smoothness(board: Sequence[int]) -> float:
        # Higher means neighboring tiles have similar exponents, a useful
        # proxy for preserving merge opportunities.
        total = 0.0
        for row in range(4):
            for col in range(4):
                value = board[row * 4 + col]
                if not value:
                    continue
                exponent = math.log2(value)
                for dr, dc in ((1, 0), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if nr < 4 and nc < 4 and board[nr * 4 + nc]:
                        total -= abs(exponent - math.log2(board[nr * 4 + nc]))
        return total


def play_heuristic_game(seed: int = 2048, max_steps: int = 5000) -> dict:
    """Play one non-learning game and return its full move history."""
    game = Game2048(random.Random(seed))
    initial_board = game.board.copy()
    moves: list[dict] = []
    while not game.is_game_over() and game.steps < max_steps:
        action = game.heuristic_action()
        if action is None:
            break
        before = game.board.copy()
        result = game.step(action)
        moves.append({**result.to_dict(), "board_before": before})
    return {**game.as_dict(), "initial_board": initial_board, "moves": moves, "policy": "heuristic-demo"}
