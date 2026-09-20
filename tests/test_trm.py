import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.game import Game2048
from backend.model import LatentReasoningPolicy, ModelConfig
from backend.trainer import TRMTrainer, TrainerConfig


class TRMTests(unittest.TestCase):
    def test_game_merge_and_observation(self) -> None:
        game = Game2048()
        game.reset([2, 2, 2, 2] + [0] * 12)
        result = game.step("left")
        self.assertTrue(result.changed)
        self.assertEqual(result.score_delta, 8)
        self.assertEqual(game.board[:2], [4, 4])
        self.assertEqual(game.observation().shape, (16,))

    def test_latent_model_update_is_finite(self) -> None:
        model = LatentReasoningPolicy(ModelConfig(hidden_size=12, reasoning_steps=3, seed=3))
        observations = np.random.default_rng(4).random((8, 16), dtype=np.float32)
        metrics = model.train_batch(observations, np.arange(8) % 4, np.linspace(-1, 3, 8))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        probabilities, values = model.predict(observations)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_demo_cycle_persists_stats_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = TRMTrainer(
                TrainerConfig(
                    demo_games=1,
                    max_steps_per_episode=30,
                    episodes_per_cycle=1,
                    updates_per_cycle=1,
                    stats_path=str(root / "stats.json"),
                    checkpoint_dir=str(root / "checkpoints"),
                ),
                load_checkpoint=False,
            )
            self.assertTrue(trainer.status()["demo_mode"])
            status = trainer.run_cycle()
            self.assertEqual(status["episodes"], 1)
            self.assertFalse(status["demo_mode"])
            trainer.save_checkpoint()
            self.assertTrue((root / "stats.json").exists())
            self.assertTrue((root / "checkpoints" / "latest.npz").exists())
            self.assertEqual(json.loads((root / "stats.json").read_text())["episodes"], 1)


if __name__ == "__main__":
    unittest.main()

