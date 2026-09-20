import json
import threading
import unittest
from http.client import HTTPConnection

from backend.server import create_server


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server("127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self) -> None:
        self.server.service.trainer.stop(wait_seconds=5)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        connection = HTTPConnection(self.host, self.port, timeout=10)
        encoded = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if encoded else {}
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    def test_state_replay_and_live_training_metrics(self) -> None:
        status, state = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(len(state["board"]), 4)
        status, stats = self.request("GET", "/api/train/stats")
        self.assertEqual(status, 200)
        self.assertIn("average_highest_tile", stats)
        status, replay = self.request("GET", "/api/replay?source=training")
        self.assertEqual(status, 200)
        self.assertIn("board", replay["replay"])

    def test_interactive_move_endpoint(self) -> None:
        status, moved = self.request("POST", "/api/move", {"action": "left"})
        self.assertEqual(status, 200)
        self.assertIn("move", moved)


if __name__ == "__main__":
    unittest.main()

