import unittest

from backend.planner import _native_library, play_expectimax_game


@unittest.skipUnless(_native_library() is not None, "the optional native planner is unavailable")
class PlannerTests(unittest.TestCase):
    def test_target_seed_reaches_4096(self) -> None:
        result = play_expectimax_game(seed=2059, max_steps=2_500, target_tile=4_096, depth=3)
        self.assertTrue(result["target_reached"])
        self.assertGreaterEqual(result["max_tile"], 4_096)


if __name__ == "__main__":
    unittest.main()
