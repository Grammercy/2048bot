"""Replay helpers for saving and restoring complete deterministic games."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .engine import GameState


def serialize_game(game: GameState, *, indent: int | None = None) -> str:
    """Serialize state, RNG stream, and all attempted turns to JSON."""

    return json.dumps(game.to_dict(include_history=True), indent=indent, separators=None if indent else (",", ":"))


def deserialize_game(value: str | Mapping[str, Any]) -> GameState:
    """Restore a replay from JSON text or an already parsed object."""

    payload = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(payload, dict):
        raise ValueError("a replay must be a JSON object")
    return GameState.from_dict(payload)


def replay_summary(game: GameState) -> dict[str, Any]:
    """Small metadata payload suitable for a game list in the frontend."""

    return {
        "seed": game.seed,
        "score": game.score,
        "moves": game.moves,
        "highest_tile": game.highest_tile,
        "won": game.won,
        "done": game.done,
        "turns": len(game.history),
    }


__all__ = ["deserialize_game", "replay_summary", "serialize_game"]
