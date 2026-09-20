"""Dependency-free self-play learner and training monitor.

This is intentionally a small, inspectable learner rather than a framework
adapter.  ``LatentQAgent`` uses a linear value model over board features and
does a few latent refinement passes before selecting an action.  The model is
simple enough to train in a background thread and to save as JSON, while the
interfaces leave room for a larger TRM model later.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .engine import Board, Direction, GameState, legal_moves, max_tile, move_board

ACTION_COUNT = len(Direction)
FEATURE_NAMES = (
    "bias",
    "empty_ratio",
    "max_log2",
    "score_potential",
    "smoothness",
    "monotonicity",
    "corner_max",
    "open_edges",
)


def _log_tile(value: int) -> float:
    return math.log2(value) if value else 0.0


def board_features(board: Board) -> tuple[float, ...]:
    """Compact features used by the baseline latent value model."""

    logs = [[_log_tile(value) for value in row] for row in board]
    maximum = max_tile(board)
    empty_ratio = sum(value == 0 for row in board for value in row) / 16.0
    # Higher is better: neighbouring tiles of similar magnitude are easier to merge.
    smoothness = 0.0
    for row in range(4):
        for column in range(4):
            if board[row][column]:
                if row < 3 and board[row + 1][column]:
                    smoothness -= abs(logs[row][column] - logs[row + 1][column])
                if column < 3 and board[row][column + 1]:
                    smoothness -= abs(logs[row][column] - logs[row][column + 1])
    # Reward rows/columns that trend in one direction, but retain scale.
    monotonicity = 0.0
    for line in logs + [list(column) for column in zip(*logs)]:
        forward = sum(line[index] - line[index + 1] for index in range(3))
        backward = -forward
        monotonicity += max(forward, backward)
    corner_max = 1.0 if maximum and maximum in (board[0][0], board[0][3], board[3][0], board[3][3]) else 0.0
    open_edges = sum(
        board[row][column] == 0
        for row in range(4)
        for column in range(4)
        if row in (0, 3) or column in (0, 3)
    ) / 12.0
    score_potential = sum(value for row in board for value in row) / (16.0 * max(2, maximum))
    return (
        1.0,
        empty_ratio,
        _log_tile(maximum) / 16.0,
        score_potential,
        smoothness / 32.0,
        monotonicity / 32.0,
        corner_max,
        open_edges,
    )


@dataclass
class LinearQModel:
    """A tiny action-value approximator that serializes to plain JSON."""

    weights: list[list[float]] = field(default_factory=lambda: [[0.0] * len(FEATURE_NAMES) for _ in Direction])
    learning_rate: float = 0.035

    def __post_init__(self) -> None:
        expected = (ACTION_COUNT, len(FEATURE_NAMES))
        if len(self.weights) != expected[0] or any(len(row) != expected[1] for row in self.weights):
            raise ValueError(f"weights must have shape {expected}")

    def q_value(self, board: Board, action: Direction | int) -> float:
        features = board_features(board)
        row = self.weights[int(Direction.parse(action))]
        return sum(weight * feature for weight, feature in zip(row, features))

    def values(self, board: Board, actions: tuple[Direction, ...] | None = None) -> dict[Direction, float]:
        actions = actions if actions is not None else tuple(Direction)
        return {action: self.q_value(board, action) for action in actions}

    def update(self, board: Board, action: Direction, target: float) -> float:
        features = board_features(board)
        prediction = self.q_value(board, action)
        error = target - prediction
        row = self.weights[int(action)]
        for index, feature in enumerate(features):
            row[index] += self.learning_rate * error * feature
        return error

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearQModel":
        return cls(
            weights=[[float(value) for value in row] for row in payload["weights"]],
            learning_rate=float(payload.get("learning_rate", 0.035)),
        )


class LatentQAgent:
    """Epsilon-greedy Q learner with iterative latent action refinement.

    ``reasoning_steps`` makes action selection consider the value of the
    resulting board repeatedly.  It is deliberately exposed as a parameter,
    so a future recurrent/TRM value head can use the same trainer API.
    """

    def __init__(
        self,
        model: LinearQModel | None = None,
        *,
        gamma: float = 0.985,
        epsilon: float = 0.14,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        reasoning_steps: int = 3,
        rng: random.Random | None = None,
    ) -> None:
        self.model = model or LinearQModel()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reasoning_steps = max(1, int(reasoning_steps))
        self.rng = rng or random.Random()
        self.updates = 0

    def _latent_values(self, board: Board, actions: tuple[Direction, ...]) -> dict[Direction, float]:
        values = self.model.values(board, actions)
        # A cheap deterministic lookahead: reward the value of each post-slide
        # board.  The repeated update is the latent reasoning loop.
        for _ in range(self.reasoning_steps):
            refined: dict[Direction, float] = {}
            for action in actions:
                result = move_board(board, action)
                next_actions = legal_moves(result.board)
                continuation = max((self.model.q_value(result.board, candidate) for candidate in next_actions), default=0.0)
                refined[action] = values[action] + self.gamma * 0.18 * continuation
            values = refined
        return values

    def select_action(self, state: GameState, *, explore: bool = True) -> Direction:
        actions = state.legal_moves
        if not actions:
            raise RuntimeError("cannot select an action from a terminal state")
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        values = self._latent_values(state.board, actions)
        best = max(values.values())
        # Random tie breaking prevents directional bias when the model is new.
        choices = [action for action, value in values.items() if abs(value - best) < 1e-12]
        return self.rng.choice(choices)

    def update(self, board: Board, action: Direction, reward: float, next_board: Board, done: bool) -> float:
        next_actions = legal_moves(next_board)
        next_value = max((self.model.q_value(next_board, candidate) for candidate in next_actions), default=0.0)
        target = reward if done else reward + self.gamma * next_value
        error = self.model.update(board, action, target)
        self.updates += 1
        return error

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "latent-q",
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "reasoning_steps": self.reasoning_steps,
            "updates": self.updates,
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentQAgent":
        agent = cls(
            model=LinearQModel.from_dict(payload["model"]),
            gamma=float(payload.get("gamma", 0.985)),
            epsilon=float(payload.get("epsilon", 0.14)),
            epsilon_min=float(payload.get("epsilon_min", 0.02)),
            epsilon_decay=float(payload.get("epsilon_decay", 0.9995)),
            reasoning_steps=int(payload.get("reasoning_steps", 3)),
        )
        agent.updates = int(payload.get("updates", 0))
        return agent


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    score: int
    moves: int
    highest_tile: int
    won: bool
    epsilon: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "score": self.score,
            "moves": self.moves,
            "highest_tile": self.highest_tile,
            "won": self.won,
            "epsilon": self.epsilon,
            "duration_ms": self.duration_ms,
        }


class SelfPlayTrainer:
    """Runs episodes and exposes thread-safe metrics for a monitoring UI."""

    def __init__(self, agent: LatentQAgent | None = None, *, seed: int | None = 7) -> None:
        self.agent = agent or LatentQAgent(rng=random.Random(seed))
        self.seed = seed
        self._episode_seed = random.Random(seed)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._recent: deque[EpisodeSummary] = deque(maxlen=60)
        self._all_time_best_tile = 0
        self._all_time_best_score = 0
        self._tile_counts: Counter[int] = Counter()
        self._games = 0
        self._total_score = 0
        self._total_moves = 0
        self._total_highest_tile = 0
        self._last_error = 0.0
        self._last_replay: GameState | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_episode(self, *, explore: bool = True, seed: int | None = None, max_moves: int = 20_000) -> tuple[GameState, EpisodeSummary]:
        started = time.perf_counter()
        state = GameState.new(seed if seed is not None else self._episode_seed.randrange(2**63))
        while not state.done and state.moves < max_moves:
            board = state.board
            action = self.agent.select_action(state, explore=explore)
            result = state.step(action)
            if explore:
                self._last_error = self.agent.update(board, action, result.reward, state.board, state.done)
        self.agent.end_episode()
        summary = EpisodeSummary(
            episode=self._games + 1,
            score=state.score,
            moves=state.moves,
            highest_tile=state.highest_tile,
            won=state.won,
            epsilon=self.agent.epsilon,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        if explore:
            with self._lock:
                self._games += 1
                self._total_score += state.score
                self._total_moves += state.moves
                self._total_highest_tile += state.highest_tile
                self._all_time_best_tile = max(self._all_time_best_tile, state.highest_tile)
                self._all_time_best_score = max(self._all_time_best_score, state.score)
                self._tile_counts[state.highest_tile] += 1
                self._recent.append(summary)
                self._last_replay = state
        return state, summary

    def _worker(self, games: int | None) -> None:
        completed = 0
        while not self._stop_event.is_set() and (games is None or completed < games):
            self.run_episode(explore=True)
            completed += 1

    def start(self, games: int | None = None) -> bool:
        """Start background training; return False if already running."""

        with self._lock:
            if self.running:
                return False
            self._stop_event.clear()
            self._started_at = time.perf_counter()
            self._thread = threading.Thread(target=self._worker, args=(games,), name="2048-self-play", daemon=True)
            self._thread.start()
            return True

    def stop(self, *, wait: bool = False) -> bool:
        with self._lock:
            was_running = self.running
            self._stop_event.set()
            thread = self._thread
        if wait and thread is not None:
            thread.join(timeout=5)
        return was_running

    def recent_games(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [summary.to_dict() for summary in list(self._recent)[-max(0, limit):]]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(0.001, time.perf_counter() - self._started_at) if self._started_at else 0.001
            recent = list(self._recent)
            recent_tiles = [summary.highest_tile for summary in recent]
            return {
                "running": self.running,
                "games": self._games,
                "updates": self.agent.updates,
                "epsilon": self.agent.epsilon,
                "episodes_per_second": self._games / elapsed,
                "average_score": self._total_score / self._games if self._games else 0.0,
                "average_moves": self._total_moves / self._games if self._games else 0.0,
                "average_highest_tile": self._total_highest_tile / self._games if self._games else 0.0,
                "average_highest_tile_recent": sum(recent_tiles) / len(recent_tiles) if recent_tiles else 0.0,
                "best_tile": self._all_time_best_tile,
                "best_score": self._all_time_best_score,
                "recent_highest_tiles": recent_tiles,
                "tile_distribution": {str(tile): count for tile, count in sorted(self._tile_counts.items())},
                "recent_games": [summary.to_dict() for summary in recent[-20:]],
                "last_td_error": self._last_error,
                "algorithm": "latent-q",
                "reasoning_steps": self.agent.reasoning_steps,
            }

    def model_dict(self) -> dict[str, Any]:
        with self._lock:
            return self.agent.to_dict()

    def last_replay(self) -> dict[str, Any] | None:
        with self._lock:
            return self._last_replay.to_dict() if self._last_replay else None

    def evaluate(self, games: int = 10) -> dict[str, Any]:
        games = max(1, min(int(games), 500))
        episodes: list[EpisodeSummary] = []
        for _ in range(games):
            _, summary = self.run_episode(explore=False)
            episodes.append(summary)
        tiles = [summary.highest_tile for summary in episodes]
        return {
            "games": games,
            "average_score": sum(item.score for item in episodes) / games,
            "average_moves": sum(item.moves for item in episodes) / games,
            "average_highest_tile": sum(tiles) / games,
            "best_tile": max(tiles),
            "wins": sum(item.won for item in episodes),
            "episodes": [item.to_dict() for item in episodes],
        }


__all__ = [
    "EpisodeSummary",
    "FEATURE_NAMES",
    "LatentQAgent",
    "LinearQModel",
    "SelfPlayTrainer",
    "board_features",
]
