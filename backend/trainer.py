"""Self-play trainer and persistence for the latent-reasoning 2048 agent."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any, Iterable

import numpy as np

from .game import ACTIONS, Game2048, play_heuristic_game
from .model import LatentReasoningPolicy, ModelConfig


@dataclass
class TrainerConfig:
    episodes_per_cycle: int = 8
    updates_per_cycle: int = 8
    batch_size: int = 64
    replay_capacity: int = 40_000
    gamma: float = 0.995
    max_steps_per_episode: int = 2_000
    initial_epsilon: float = 0.22
    final_epsilon: float = 0.025
    epsilon_decay_episodes: int = 1_500
    temperature: float = 0.85
    checkpoint_every_episodes: int = 100
    stats_path: str = "data/training_stats.json"
    checkpoint_dir: str = "data/checkpoints"
    seed: int = 2048
    demo_games: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    return_: float


class ReplayBuffer:
    """A bounded replay buffer storing complete, discounted self-play targets."""

    def __init__(self, capacity: int = 40_000, seed: int = 2048) -> None:
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    def add_episode(self, observations: Iterable[np.ndarray], actions: Iterable[int], rewards: Iterable[float], gamma: float) -> None:
        obs = list(observations)
        acts = list(actions)
        rewards = list(rewards)
        returns = [0.0] * len(rewards)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = float(rewards[index]) + gamma * running
            returns[index] = running
        for observation, action, target in zip(obs, acts, returns):
            self._items.append(Transition(np.asarray(observation, dtype=np.float32).copy(), int(action), float(target)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._items:
            raise ValueError("cannot sample an empty replay buffer")
        count = min(int(batch_size), len(self._items))
        batch = self.rng.sample(list(self._items), count)
        return (
            np.stack([item.observation for item in batch]).astype(np.float32),
            np.asarray([item.action for item in batch], dtype=np.int64),
            np.asarray([item.return_ for item in batch], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class Metrics:
    episodes: int = 0
    steps: int = 0
    best_tile: int = 0
    mean_max_tile: float = 0.0
    best_score: int = 0
    mean_score: float = 0.0
    mean_episode_length: float = 0.0
    replay_size: int = 0
    updates: int = 0
    last_policy_loss: float = 0.0
    last_value_loss: float = 0.0
    last_entropy: float = 0.0
    last_update_seconds: float = 0.0
    training_seconds: float = 0.0
    demo_mode: bool = True
    last_error: str | None = None
    updated_at: float = field(default_factory=time.time)


class TRMTrainer:
    """Thread-safe trainer facade used by both scripts and the monitoring API."""

    def __init__(self, config: TrainerConfig | None = None, *, load_checkpoint: bool = True) -> None:
        self.config = config or TrainerConfig()
        # Keep model and trainer seeds related but distinct so board randomness
        # is not coupled to weight initialization.
        model_config = self.config.model
        self.model = LatentReasoningPolicy(model_config)
        self.rng = random.Random(self.config.seed)
        self.replay = ReplayBuffer(self.config.replay_capacity, self.config.seed + 1)
        self.metrics = Metrics()
        self._recent_tiles: deque[int] = deque(maxlen=100)
        self._recent_scores: deque[int] = deque(maxlen=100)
        self._recent_lengths: deque[int] = deque(maxlen=100)
        self._latest_game: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._status = "idle"
        loaded_state = False
        if load_checkpoint:
            loaded_state = self._load_latest_if_present()
            if not loaded_state:
                loaded_state = self._load_stats_if_present()
        # A few deterministic heuristic games make a freshly started dashboard
        # useful, while being explicit that these are demo (not learned) stats.
        if self.metrics.episodes == 0 and not loaded_state:
            self.seed_demo(self.config.demo_games)
        elif loaded_state:
            self.metrics.demo_mode = False
        self._persist_stats()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def seed_demo(self, games: int = 3) -> None:
        demos: list[dict[str, Any]] = []
        for offset in range(max(1, int(games))):
            demos.append(play_heuristic_game(self.config.seed + offset, self.config.max_steps_per_episode))
        with self._lock:
            latest = max(demos, key=lambda item: (item.get("max_tile", 0), item.get("score", 0)))
            self._latest_game = latest
            self.metrics.demo_mode = True
            self.metrics.best_tile = max(int(item["max_tile"]) for item in demos)
            self.metrics.mean_max_tile = float(np.mean([item["max_tile"] for item in demos]))
            self.metrics.best_score = max(int(item["score"]) for item in demos)
            self.metrics.mean_score = float(np.mean([item["score"] for item in demos]))
            self.metrics.mean_episode_length = float(np.mean([item["steps"] for item in demos]))
            self.metrics.updated_at = time.time()

    def run_cycle(self, *, episodes: int | None = None, updates: int | None = None) -> dict[str, Any]:
        """Run a bounded amount of work synchronously (useful in tests/CLI)."""
        if self.running and threading.current_thread() is not self._thread:
            return self.status()
        started = time.perf_counter()
        with self._lock:
            self._status = "training"
            self.metrics.demo_mode = False
            self.metrics.last_error = None
        episode_count = self.config.episodes_per_cycle if episodes is None else max(0, int(episodes))
        update_count = self.config.updates_per_cycle if updates is None else max(0, int(updates))
        for _ in range(episode_count):
            if self._stop_event.is_set():
                break
            self._play_episode()
        update_metrics: dict[str, float] = {}
        for _ in range(update_count):
            if self._stop_event.is_set() or len(self.replay) < 1:
                break
            update_metrics = self._train_update()
        elapsed = time.perf_counter() - started
        with self._lock:
            self.metrics.training_seconds += elapsed
            self.metrics.replay_size = len(self.replay)
            self.metrics.updated_at = time.time()
            self._status = "stopping" if self._stop_event.is_set() else "idle"
            should_checkpoint = (
                self.metrics.episodes > 0
                and self.config.checkpoint_every_episodes > 0
                and self.metrics.episodes % self.config.checkpoint_every_episodes < max(1, episode_count)
            )
            snapshot = self._status_unlocked()
            self._persist_stats_unlocked()
            if should_checkpoint:
                self._save_checkpoint_unlocked()
        return snapshot

    def start_async(self, *, episodes: int | None = None, updates: int | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop_event.clear()
            self._started_at = time.time()
            self._status = "training"

        def worker() -> None:
            try:
                while not self._stop_event.is_set():
                    self.run_cycle(episodes=episodes, updates=updates)
            except Exception as exc:  # surface errors to the dashboard and caller
                with self._lock:
                    self.metrics.last_error = f"{type(exc).__name__}: {exc}"
                    self._status = "error"
                    self._persist_stats_unlocked()
            finally:
                with self._lock:
                    if self._status != "error":
                        self._status = "idle"
                    self._persist_stats_unlocked()

        self._thread = threading.Thread(target=worker, name="trm-self-play", daemon=True)
        self._thread.start()
        return True

    def stop(self, wait_seconds: float = 3.0) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, wait_seconds))
        with self._lock:
            was_running = bool(thread and thread.is_alive())
            if not was_running and self._status != "error":
                self._status = "idle"
            self._persist_stats_unlocked()
        return not was_running

    def play_game(
        self,
        *,
        greedy: bool = True,
        max_steps: int | None = None,
        policy: str = "trm",
    ) -> dict[str, Any]:
        """Play with the selected policy and return move-by-move board states.

        The recurrent model remains the default training/evaluation policy.
        ``policy="expectimax"`` exposes the stronger deterministic evaluator
        used when a run has an explicit target tile.
        """

        if policy.strip().lower() in {"expectimax", "planner", "strong"}:
            from .planner import play_expectimax_game

            return play_expectimax_game(
                seed=2059,
                max_steps=max_steps or 5_000,
                target_tile=4_096,
                depth=3,
            )
        game = Game2048(random.Random(self.rng.randrange(2**31)))
        initial_board = game.board.copy()
        moves: list[dict] = []
        limit = max_steps or self.config.max_steps_per_episode
        while not game.is_game_over() and game.steps < limit:
            legal = game.legal_actions()
            if not legal:
                break
            probabilities, _ = self.model.predict(game.observation())
            action = self._choose_action(probabilities[0], legal, epsilon=0.0 if greedy else self.config.final_epsilon)
            before = game.board.copy()
            result = game.step(action)
            moves.append({**result.to_dict(), "board_before": before})
        result = game.as_dict()
        result["initial_board"] = initial_board
        result["moves"] = moves
        result["policy"] = "trm"
        return result

    def play_strong_game(self, *, max_steps: int = 5_000) -> dict[str, Any]:
        """Run the target-seeking expectimax evaluator."""

        return self.play_game(greedy=True, max_steps=max_steps, policy="expectimax")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def latest_game(self) -> dict[str, Any] | None:
        with self._lock:
            return json.loads(json.dumps(self._latest_game)) if self._latest_game is not None else None

    def _play_episode(self) -> None:
        game = Game2048(random.Random(self.rng.randrange(2**31)))
        initial_board = game.board.copy()
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        moves: list[dict] = []
        epsilon = self._epsilon()
        for _ in range(self.config.max_steps_per_episode):
            legal = game.legal_actions()
            if not legal:
                break
            observation = game.observation()
            probabilities, _ = self.model.predict(observation)
            action = self._choose_action(probabilities[0], legal, epsilon=epsilon)
            before = game.board.copy()
            result = game.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(result.reward)
            moves.append({**result.to_dict(), "board_before": before})
            if result.done:
                break
        self.replay.add_episode(observations, actions, rewards, self.config.gamma)
        game_result = game.as_dict()
        game_result["initial_board"] = initial_board
        game_result["moves"] = moves
        game_result["policy"] = "trm"
        with self._lock:
            self._latest_game = game_result
            self.metrics.episodes += 1
            self.metrics.steps += game.steps
            self._recent_tiles.append(game.max_tile)
            self._recent_scores.append(game.score)
            self._recent_lengths.append(game.steps)
            self.metrics.best_tile = max(self.metrics.best_tile, game.max_tile)
            self.metrics.best_score = max(self.metrics.best_score, game.score)
            self.metrics.mean_max_tile = float(np.mean(self._recent_tiles))
            self.metrics.mean_score = float(np.mean(self._recent_scores))
            self.metrics.mean_episode_length = float(np.mean(self._recent_lengths))
            self.metrics.replay_size = len(self.replay)
            self.metrics.updated_at = time.time()

    def _train_update(self) -> dict[str, float]:
        observations, actions, returns = self.replay.sample(self.config.batch_size)
        started = time.perf_counter()
        result = self.model.train_batch(observations, actions, returns)
        result["duration"] = time.perf_counter() - started
        with self._lock:
            self.metrics.updates += 1
            self.metrics.last_policy_loss = result["policy_loss"]
            self.metrics.last_value_loss = result["value_loss"]
            self.metrics.last_entropy = result["entropy"]
            self.metrics.last_update_seconds = result["duration"]
            self.metrics.updated_at = time.time()
        return result

    def _choose_action(self, probabilities: np.ndarray, legal: list[int], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)
        weights = np.asarray([max(0.0, float(probabilities[action])) for action in legal], dtype=np.float64)
        temperature = max(0.05, self.config.temperature)
        if temperature != 1.0:
            weights = np.power(weights + 1e-8, 1.0 / temperature)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0:
            return self.rng.choice(legal)
        weights /= total
        return int(self.rng.choices(legal, weights=weights.tolist(), k=1)[0])

    def _epsilon(self) -> float:
        progress = min(1.0, self.metrics.episodes / max(1, self.config.epsilon_decay_episodes))
        return self.config.initial_epsilon + (self.config.final_epsilon - self.config.initial_epsilon) * progress

    def _status_unlocked(self) -> dict[str, Any]:
        payload = asdict(self.metrics)
        payload.update(
            {
                "status": self._status,
                "running": self.running,
                "epsilon": self._epsilon(),
                "model_step": self.model.state_dict()["step"],
                "reasoning_steps": self.model.config.reasoning_steps,
                "hidden_size": self.model.config.hidden_size,
                "started_at": self._started_at,
                "recent_max_tiles": list(self._recent_tiles),
                "recent_scores": list(self._recent_scores),
                "recent_lengths": list(self._recent_lengths),
                "latest_game_available": self._latest_game is not None,
            }
        )
        return _jsonable(payload)

    def _persist_stats(self) -> None:
        with self._lock:
            self._persist_stats_unlocked()

    def _persist_stats_unlocked(self) -> None:
        path = Path(self.config.stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._status_unlocked(), indent=2), encoding="utf-8")
        tmp.replace(path)

    def _save_checkpoint_unlocked(self) -> Path:
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "latest.npz"
        self.model.save(model_path)
        (directory / "latest.json").write_text(json.dumps(self._status_unlocked(), indent=2), encoding="utf-8")
        return model_path

    def save_checkpoint(self) -> str:
        with self._lock:
            return str(self._save_checkpoint_unlocked())

    def _load_latest_if_present(self) -> bool:
        path = Path(self.config.checkpoint_dir) / "latest.npz"
        if not path.exists():
            return False
        try:
            self.model = LatentReasoningPolicy.load(path)
            self.metrics.demo_mode = False
            metadata_path = path.with_suffix(".json")
            if metadata_path.exists():
                self._restore_metrics(json.loads(metadata_path.read_text(encoding="utf-8")))
            return True
        except Exception as exc:
            # A partial/corrupt checkpoint should not make the dashboard fail
            # to start; the error remains visible in the status endpoint.
            self.metrics.last_error = f"checkpoint load failed: {exc}"
            return False

    def _load_stats_if_present(self) -> bool:
        path = Path(self.config.stats_path)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return False
            self._restore_metrics(payload)
            return self.metrics.episodes > 0
        except (OSError, ValueError, TypeError):
            return False

    def _restore_metrics(self, payload: dict[str, Any]) -> None:
        metric_names = {item.name for item in fields(Metrics)}
        for name in metric_names:
            if name not in payload:
                continue
            try:
                setattr(self.metrics, name, payload[name])
            except (TypeError, ValueError):
                continue
        self._recent_tiles.extend(int(value) for value in payload.get("recent_max_tiles", []))
        self._recent_scores.extend(int(value) for value in payload.get("recent_scores", []))
        self._recent_lengths.extend(int(value) for value in payload.get("recent_lengths", []))
        self._status = "idle"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
