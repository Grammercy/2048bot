"""TRM-style recurrent policy-gradient self-play for 2048.

This trainer uses :class:`LatentReasoningPolicy` from ``model.py``. The same
latent transition is reused for several internal reasoning steps before the
policy chooses a move. It is deliberately small enough to run on CPU while
still exposing real policy/value losses and entropy to the monitor.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .game import ACTIONS, Game2048
from .model import LatentReasoningPolicy, ModelConfig


@dataclass(frozen=True)
class TRMEpisodeSummary:
    episode: int
    score: int
    moves: int
    highest_tile: int
    won: bool
    epsilon: float
    entropy: float
    policy_loss: float
    value_loss: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "score": self.score,
            "moves": self.moves,
            "highest_tile": self.highest_tile,
            "won": self.won,
            "epsilon": round(self.epsilon, 6),
            "entropy": round(self.entropy, 6),
            "policy_loss": round(self.policy_loss, 6),
            "value_loss": round(self.value_loss, 6),
            "duration_ms": round(self.duration_ms, 3),
        }


class TRMSelfPlayTrainer:
    """Threaded self-play trainer with a polling-friendly metrics snapshot."""

    def __init__(self, *, seed: int = 7, model: LatentReasoningPolicy | None = None) -> None:
        self.seed = seed
        self.model = model or LatentReasoningPolicy(ModelConfig(seed=seed, reasoning_steps=4))
        self.rng = random.Random(seed)
        self._episode_seed = random.Random(seed)
        self.epsilon = 0.18
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.9992
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._recent: deque[TRMEpisodeSummary] = deque(maxlen=60)
        self._games = 0
        self._total_score = 0
        self._total_moves = 0
        self._best_tile = 0
        self._best_score = 0
        self._last_replay: dict[str, Any] | None = None
        self._last_metrics = {"entropy": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _select(self, game: Game2048, *, explore: bool) -> tuple[int, float]:
        legal = game.legal_actions()
        probs, _ = self.model.predict(game.observation())
        if explore and self.rng.random() < self.epsilon:
            action = self.rng.choice(legal)
        else:
            action = max(legal, key=lambda candidate: float(probs[candidate]))
        return action, float(probs[action])

    @staticmethod
    def _board_rows(flat: list[int] | tuple[int, ...]) -> list[list[int]]:
        return [list(flat[index : index + 4]) for index in range(0, 16, 4)]

    def run_episode(self, *, explore: bool = True, seed: int | None = None, max_moves: int = 5000) -> tuple[dict[str, Any], TRMEpisodeSummary]:
        started = time.perf_counter()
        game = Game2048(random.Random(seed if seed is not None else self._episode_seed.randrange(2**63)))
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        history: list[dict[str, Any]] = []
        while not game.is_game_over() and game.steps < max_moves:
            before = game.board.copy()
            observation = game.observation()
            action, _ = self._select(game, explore=explore)
            result = game.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(result.reward)
            history.append({
                "index": len(history),
                "action": ACTIONS[action],
                "board_before": self._board_rows(before),
                "board_after": self._board_rows(result.board),
                "changed": result.changed,
                "score_delta": result.score_delta,
                "reward": result.reward,
                "done": result.done,
            })
        metrics = {"entropy": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        if explore and observations:
            returns = np.zeros(len(rewards), dtype=np.float32)
            running = 0.0
            for index in range(len(rewards) - 1, -1, -1):
                running = rewards[index] + 0.985 * running
                returns[index] = running
            metrics = self.model.train_batch(np.asarray(observations), np.asarray(actions), returns)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        summary = TRMEpisodeSummary(
            episode=self._games + 1,
            score=game.score,
            moves=game.steps,
            highest_tile=game.max_tile,
            won=game.max_tile >= 2048,
            epsilon=self.epsilon,
            entropy=metrics.get("entropy", 0.0),
            policy_loss=metrics.get("policy_loss", 0.0),
            value_loss=metrics.get("value_loss", 0.0),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        replay = {
            "board": self._board_rows(game.board),
            "score": game.score,
            "moves": game.steps,
            "highest_tile": game.max_tile,
            "won": summary.won,
            "done": game.is_game_over(),
            "history": history,
        }
        if explore:
            with self._lock:
                self._games += 1
                self._total_score += game.score
                self._total_moves += game.steps
                self._best_tile = max(self._best_tile, game.max_tile)
                self._best_score = max(self._best_score, game.score)
                self._recent.append(summary)
                self._last_replay = replay
                self._last_metrics = metrics
        return replay, summary

    def _worker(self, games: int | None) -> None:
        complete = 0
        while not self._stop_event.is_set() and (games is None or complete < games):
            self.run_episode(explore=True)
            complete += 1

    def start(self, games: int | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop_event.clear()
            self._started_at = time.perf_counter()
            self._thread = threading.Thread(target=self._worker, args=(games,), daemon=True, name="trm-self-play")
            self._thread.start()
            return True

    def stop(self, *, wait: bool = False) -> bool:
        with self._lock:
            was_running = self.running
            self._stop_event.set()
            thread = self._thread
        if wait and thread is not None:
            thread.join(timeout=10)
        return was_running

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(0.001, time.perf_counter() - self._started_at) if self._started_at else 0.001
            recent = list(self._recent)
            tiles = [entry.highest_tile for entry in recent]
            return {
                "running": self.running,
                "games": self._games,
                "updates": int(self.model._step),
                "epsilon": self.epsilon,
                "episodes_per_second": self._games / elapsed,
                "average_score": self._total_score / self._games if self._games else 0.0,
                "average_moves": self._total_moves / self._games if self._games else 0.0,
                "average_highest_tile": sum(tiles) / len(tiles) if tiles else 0.0,
                "best_tile": self._best_tile,
                "best_score": self._best_score,
                "recent_highest_tiles": tiles,
                "recent_games": [entry.to_dict() for entry in recent[-20:]],
                "last_td_error": self._last_metrics.get("value_loss", 0.0),
                "entropy": self._last_metrics.get("entropy", 0.0),
                "policy_loss": self._last_metrics.get("policy_loss", 0.0),
                "value_loss": self._last_metrics.get("value_loss", 0.0),
                "algorithm": "trm-latent-policy",
                "reasoning_steps": self.model.config.reasoning_steps,
            }

    def recent_games(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [entry.to_dict() for entry in list(self._recent)[-max(0, limit):]]

    def last_replay(self) -> dict[str, Any] | None:
        with self._lock:
            return self._last_replay

    def model_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "trm-latent-policy",
            "config": self.model.config.__dict__,
            "optimizer_step": int(self.model._step),
            "parameter_count": int(sum(array.size for array in self.model.params.values())),
        }

    def evaluate(self, games: int = 10) -> dict[str, Any]:
        games = max(1, min(int(games), 100))
        episodes = [self.run_episode(explore=False)[1] for _ in range(games)]
        tiles = [episode.highest_tile for episode in episodes]
        return {
            "games": games,
            "average_score": sum(entry.score for entry in episodes) / games,
            "average_moves": sum(entry.moves for entry in episodes) / games,
            "average_highest_tile": sum(tiles) / games,
            "best_tile": max(tiles),
            "wins": sum(entry.won for entry in episodes),
            "episodes": [entry.to_dict() for entry in episodes],
        }


__all__ = ["TRMEpisodeSummary", "TRMSelfPlayTrainer"]

