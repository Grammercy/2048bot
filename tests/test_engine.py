import json
import unittest

from backend.engine import Direction, GameState, legal_moves, move_board
from backend.replay import deserialize_game, serialize_game


class EngineTests(unittest.TestCase):
    def test_left_merges_once_per_tile(self) -> None:
        result = move_board(((2, 2, 2, 2), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), Direction.LEFT)
        self.assertEqual(result.board[0], (4, 4, 0, 0))
        self.assertEqual(result.score_delta, 8)
        self.assertEqual(result.merges, 2)

    def test_all_directions(self) -> None:
        board = ((2, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(move_board(board, Direction.UP).board[0][0], 4)
        self.assertEqual(move_board(board, Direction.DOWN).board[3][0], 4)
        self.assertEqual(move_board(board, Direction.RIGHT).board[0][3], 2)
        self.assertEqual(move_board(board, Direction.LEFT).board[0][0], 2)

    def test_seed_reproducibility_including_spawn(self) -> None:
        first = GameState.new(123)
        second = GameState.new(123)
        for action in (Direction.LEFT, Direction.DOWN, Direction.RIGHT, Direction.UP):
            first.step(action)
            second.step(action)
        self.assertEqual(first.board, second.board)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.to_json(), second.to_json())

    def test_legal_moves_and_terminal_board(self) -> None:
        board = ((2, 4, 8, 16), (32, 64, 128, 256), (512, 1024, 2048, 4096), (8192, 16384, 32768, 65536))
        self.assertEqual(legal_moves(board), ())
        state = GameState(board=board)
        result = state.step(Direction.LEFT)
        self.assertTrue(result.done)
        self.assertEqual(result.reward, -0.25)

    def test_replay_round_trip(self) -> None:
        game = GameState.new(7)
        game.step(Direction.RIGHT)
        game.step(Direction.DOWN)
        restored = deserialize_game(serialize_game(game))
        self.assertEqual(restored.to_dict(), game.to_dict())
        self.assertEqual(json.loads(serialize_game(game))["history"][0]["action"], "right")


if __name__ == "__main__":
    unittest.main()
