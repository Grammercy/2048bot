"""FastAPI integration for monitoring and controlling self-play."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .trainer import TRMTrainer


class TrainingStartRequest(BaseModel):
    episodes_per_cycle: int | None = Field(default=None, ge=1, le=10_000)
    updates_per_cycle: int | None = Field(default=None, ge=0, le=10_000)


class TrainingCycleRequest(BaseModel):
    episodes: int | None = Field(default=None, ge=0, le=10_000)
    updates: int | None = Field(default=None, ge=0, le=10_000)


class GameRequest(BaseModel):
    greedy: bool = True
    max_steps: int | None = Field(default=None, ge=1, le=10_000)
    policy: str = Field(default="trm", pattern="^(trm|expectimax|planner|strong)$")


def create_app(trainer: TRMTrainer | None = None) -> FastAPI:
    agent = trainer or TRMTrainer()
    app = FastAPI(title="2048 TRM Self-Play", version="1.0")
    app.state.trainer = agent
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def monitor_payload() -> dict[str, Any]:
        payload = agent.status()
        # Keep the names used by the lightweight dashboard stable even as the
        # richer trainer exposes more detailed metric names.
        payload.update(
            {
                "average_highest_tile": payload.get("mean_max_tile", 0.0),
                "average_score": payload.get("mean_score", 0.0),
                "average_moves": payload.get("mean_episode_length", 0.0),
                "games": payload.get("episodes", 0),
                "episodes_per_second": (
                    payload.get("episodes", 0) / payload["training_seconds"]
                    if payload.get("training_seconds", 0) > 0
                    else 0.0
                ),
            }
        )
        return payload

    @app.get("/api/training/status")
    @app.get("/api/training/stats")
    @app.get("/api/train/stats")
    @app.get("/api/status")
    def training_status() -> dict[str, Any]:
        return monitor_payload()

    @app.post("/api/training/start")
    @app.post("/api/train/start")
    @app.post("/api/train")
    def training_start(request: TrainingStartRequest | None = None) -> dict[str, Any]:
        request = request or TrainingStartRequest()
        started = agent.start_async(episodes=request.episodes_per_cycle, updates=request.updates_per_cycle)
        return {"started": started, **monitor_payload()}

    @app.post("/api/training/stop")
    @app.post("/api/train/stop")
    @app.post("/api/stop")
    def training_stop() -> dict[str, Any]:
        stopped = agent.stop()
        return {"stopped": stopped, **monitor_payload()}

    @app.post("/api/train/reset")
    @app.post("/api/training/reset")
    @app.post("/api/new")
    def training_reset() -> dict[str, Any]:
        agent.stop()
        # The trainer's deterministic demo seed gives the UI an immediate
        # board while leaving learned weights and checkpoints intact.
        agent.seed_demo(agent.config.demo_games)
        return {"reset": True, **monitor_payload()}

    @app.post("/api/training/cycle")
    def training_cycle(request: TrainingCycleRequest | None = None) -> dict[str, Any]:
        if agent.running:
            raise HTTPException(status_code=409, detail="training is already running asynchronously")
        request = request or TrainingCycleRequest()
        return agent.run_cycle(episodes=request.episodes, updates=request.updates)

    @app.post("/api/training/checkpoint")
    def training_checkpoint() -> dict[str, Any]:
        return {"path": agent.save_checkpoint(), **agent.status()}

    @app.get("/api/games/latest")
    def latest_game() -> dict[str, Any]:
        game = agent.latest_game()
        if game is None:
            raise HTTPException(status_code=404, detail="no game has been played yet")
        return game

    @app.get("/api/replay")
    def replay() -> dict[str, Any]:
        """Compatibility shape for the original 4x4 dashboard replay card."""
        game = agent.latest_game()
        if game is None:
            raise HTTPException(status_code=404, detail="no game has been played yet")
        flat = list(game.get("board", []))
        grid = [flat[index : index + 4] for index in range(0, 16, 4)] if len(flat) == 16 else flat
        state = {"board": grid, "score": game.get("score", 0), "steps": game.get("steps", 0)}
        return {"replay": game, "state": state, "board": grid}

    @app.get("/api/games")
    def games(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        game = agent.latest_game()
        if game is None:
            return {"games": []}
        return {
            "games": [
                {
                    "score": game.get("score", 0),
                    "moves": game.get("steps", 0),
                    "highest_tile": game.get("max_tile", 0),
                    "policy": game.get("policy", "trm"),
                }
            ][:limit]
        }

    @app.get("/api/model")
    def model_info() -> dict[str, Any]:
        return {
            "algorithm": "trm-latent-policy-value",
            "reasoning_steps": agent.model.config.reasoning_steps,
            "hidden_size": agent.model.config.hidden_size,
            "model_step": agent.model.state_dict()["step"],
        }

    @app.post("/api/games/play")
    def play_game(request: GameRequest | None = None) -> dict[str, Any]:
        request = request or GameRequest()
        return agent.play_game(greedy=request.greedy, max_steps=request.max_steps, policy=request.policy)

    return app


app = create_app()
