"""2048 latent-reasoning self-play backend.

The API factory is imported lazily so importing the game/model primitives does
not start a demo trainer or write a stats file as a side effect.
"""

from .game import ACTIONS, Game2048
from .model import LatentReasoningPolicy, ModelConfig
from .trainer import ReplayBuffer, TRMTrainer, TrainerConfig


def create_app(*args, **kwargs):
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)

__all__ = [
    "ACTIONS",
    "Game2048",
    "LatentReasoningPolicy",
    "ModelConfig",
    "ReplayBuffer",
    "TRMTrainer",
    "TrainerConfig",
    "create_app",
]
