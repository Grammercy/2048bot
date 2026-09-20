"""Small HTTP API for the 2048 self-play trainer.

Run ``python -m backend.server --port 8000`` in one terminal and the Vite UI
in another. The API is standard-library only and keeps a separate interactive
board alongside the trainer's latest self-play replay.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import zip_longest
from typing import Any
from urllib.parse import parse_qs, urlparse

from .engine import GameState
from .game import play_heuristic_game
from .trainer import TRMTrainer


HOST = os.environ.get("LATENT_2048_HOST", "127.0.0.1")
PORT = int(os.environ.get("LATENT_2048_PORT", "8000"))


def _demo_replay() -> dict[str, Any]:
    return play_heuristic_game(seed=11, max_steps=180)


@dataclass
class Service:
    """Mutable state owned by one HTTP server instance."""

    trainer: TRMTrainer
    game: GameState
    lock: threading.RLock
    last_training_replay: dict[str, Any] | None = None

    @classmethod
    def create(cls, seed: int = 7) -> "Service":
        return cls(TRMTrainer(load_checkpoint=True), GameState.new(seed=seed), threading.RLock())


def _stats(service: Service) -> dict[str, Any]:
    """Normalize the trainer's richer status for the lightweight dashboard."""
    payload = service.trainer.status()
    episodes = int(payload.get("episodes", 0))
    seconds = float(payload.get("training_seconds", 0.0))
    payload.update(
        {
            "games": episodes,
            "average_highest_tile": payload.get("mean_max_tile", 0.0),
            "average_score": payload.get("mean_score", 0.0),
            "average_moves": payload.get("mean_episode_length", 0.0),
            "episodes_per_second": episodes / seconds if seconds > 0 else 0.0,
            "algorithm": "trm-latent-policy-value",
            "recent_games": [
                {
                    "episode": max(1, episodes - len(payload.get("recent_max_tiles", [])) + index + 1),
                    "highest_tile": tile,
                    "score": score,
                    "moves": moves,
                }
                for index, (tile, score, moves) in enumerate(zip_longest(
                    payload.get("recent_max_tiles", []),
                    payload.get("recent_scores", []),
                    payload.get("recent_lengths", []),
                    fillvalue=None,
                ))
                if tile is not None
            ],
        }
    )
    return payload


def _latest_replay(service: Service) -> dict[str, Any]:
    replay = service.trainer.latest_game()
    if replay is None:
        # A restarted service can still expose an actual learned replay when
        # a checkpoint was loaded. Only a fresh trainer uses the labeled demo.
        if service.trainer.status().get("demo_mode"):
            return _demo_replay()
        if service.last_training_replay is None:
            service.last_training_replay = service.trainer.play_game(greedy=True)
        replay = service.last_training_replay
    flat = list(replay.get("board", []))
    grid = [flat[index : index + 4] for index in range(0, 16, 4)] if len(flat) == 16 else flat
    replay["board"] = grid
    if replay.get("episode") is None and not service.trainer.status().get("demo_mode"):
        replay["episode"] = service.trainer.status().get("episodes")
    return replay


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "latent-2048/0.2"

    @property
    def service(self) -> Service:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, payload: Any, status: int = 200) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            # Browser polling can cancel a request during a refresh.
            pass

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        service = self.service
        if path in {"/", "/api/health"}:
            self._send({"ok": True, "service": "latent-2048", "trainer": _stats(service)})
        elif path == "/api/state":
            with service.lock:
                state = service.game.to_dict(include_history=False)
            state["training"] = _stats(service)
            self._send(state)
        elif path in {"/api/train/stats", "/api/stats", "/api/status", "/api/training/status", "/api/training/stats"}:
            self._send(_stats(service))
        elif path == "/api/games":
            try:
                limit = int(parse_qs(parsed.query).get("limit", [20])[0])
            except ValueError:
                limit = 20
            recent = _stats(service).get("recent_max_tiles", [])[-max(1, min(limit, 100)) :]
            self._send({"games": [{"highest_tile": tile} for tile in recent]})
        elif path == "/api/replay":
            query = parse_qs(parsed.query)
            if query.get("source", [""])[0] == "training":
                replay = _latest_replay(service)
            else:
                with service.lock:
                    replay = service.game.to_dict(include_history=True)
            self._send({"replay": replay, "state": replay})
        elif path == "/api/model":
            self._send(
                {
                    "algorithm": "trm-latent-policy-value",
                    "reasoning_steps": service.trainer.model.config.reasoning_steps,
                    "hidden_size": service.trainer.model.config.hidden_size,
                    "model_step": service.trainer.model.state_dict()["step"],
                }
            )
        elif path == "/api/evaluate":
            try:
                games = int(parse_qs(parsed.query).get("games", [10])[0])
            except ValueError:
                games = 10
            policy = parse_qs(parsed.query).get("policy", ["trm"])[0]
            episodes = [service.trainer.play_game(greedy=True, policy=policy) for _ in range(max(1, min(games, 100)))]
            tiles = [int(item.get("max_tile", 0)) for item in episodes]
            self._send(
                {
                    "games": len(episodes),
                    "average_score": sum(item.get("score", 0) for item in episodes) / len(episodes),
                    "average_moves": sum(item.get("steps", 0) for item in episodes) / len(episodes),
                    "average_highest_tile": sum(tiles) / len(tiles),
                    "best_tile": max(tiles),
                    "wins": sum(tile >= 2048 for tile in tiles),
                    "episodes": episodes,
                }
            )
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        body = self._read_json()
        service = self.service
        if path in {"/api/new", "/api/reset", "/api/train/reset", "/api/training/reset"}:
            seed = body.get("seed")
            with service.lock:
                service.game.reset(None if seed is None else int(seed))
                state = service.game.to_dict(include_history=False)
            self._send({"reset": True, "state": state})
        elif path == "/api/move":
            try:
                with service.lock:
                    result = service.game.step(body.get("action", "left"))
                    state = service.game.to_dict(include_history=False)
                self._send({"move": result.to_dict(), "state": state})
            except (TypeError, ValueError) as exc:
                self._send({"error": str(exc)}, status=400)
        elif path in {"/api/train", "/api/train/start", "/api/training/start"}:
            games = body.get("games")
            try:
                games_value = None if games in (None, "", 0) else max(1, int(games))
                started = service.trainer.start_async(episodes=games_value)
            except (TypeError, ValueError) as exc:
                self._send({"error": str(exc)}, status=400)
                return
            self._send({"started": started, "state": _stats(service)})
        elif path in {"/api/train/stop", "/api/training/stop", "/api/stop", "/api/pause"}:
            stopped = service.trainer.stop()
            self._send({"stopped": stopped, "state": _stats(service)})
        elif path == "/api/evaluate":
            try:
                games = max(1, int(body.get("games", 10)))
            except (TypeError, ValueError):
                games = 10
            policy = str(body.get("policy", "trm"))
            episodes = [service.trainer.play_game(greedy=True, policy=policy) for _ in range(games)]
            tiles = [int(item.get("max_tile", 0)) for item in episodes]
            self._send({"games": games, "average_highest_tile": sum(tiles) / games, "best_tile": max(tiles), "episodes": episodes})
        else:
            self._send({"error": "not found"}, status=404)


def create_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ApiHandler)
    server.service = Service.create()  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the latent 2048 self-play API")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"latent-2048 API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.service.trainer.stop(wait_seconds=10)  # type: ignore[attr-defined]
        server.server_close()


if __name__ == "__main__":
    main()
