import unittest

from backend.engine import Direction, GameState
from backend.training import LatentQAgent, SelfPlayTrainer, board_features


class TrainingTests(unittest.TestCase):
    def test_feature_vector_and_action_selection(self) -> None:
        state = GameState.new(1)
        self.assertEqual(len(board_features(state.board)), 8)
        action = LatentQAgent().select_action(state, explore=False)
        self.assertIn(action, state.legal_moves)

    def test_short_episode_updates_model_and_stats(self) -> None:
        trainer = SelfPlayTrainer(seed=2)
        state, summary = trainer.run_episode(seed=2, max_moves=20)
        self.assertGreaterEqual(summary.moves, 0)
        self.assertEqual(summary.moves, state.moves)
        self.assertEqual(trainer.snapshot()["games"], 1)
        self.assertEqual(trainer.snapshot()["updates"], len(state.history))

    def test_finite_background_training(self) -> None:
        trainer = SelfPlayTrainer(seed=3)
        self.assertTrue(trainer.start(games=2))
        while trainer.running:
            pass
        self.assertEqual(trainer.snapshot()["games"], 2)
        self.assertFalse(trainer.start(games=1)) if trainer.running else None


if __name__ == "__main__":
    unittest.main()
