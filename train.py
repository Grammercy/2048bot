"""Run the trainer locally.

Examples:
  python train.py --episodes 100
  python train.py --serve  # requires uvicorn
"""

from __future__ import annotations

import argparse
import json

from backend.trainer import TRMTrainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2048 TRM self-play agent")
    parser.add_argument("--episodes", type=int, default=0, help="number of synchronous episodes (0 starts async)")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--serve", action="store_true", help="serve the monitoring API with uvicorn")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--evaluate-strong", action="store_true", help="run the target-seeking expectimax evaluator")
    parser.add_argument("--seed", type=int, default=2059)
    parser.add_argument("--target-tile", type=int, default=4096)
    args = parser.parse_args()
    if args.evaluate_strong:
        from backend.planner import play_expectimax_game

        result = play_expectimax_game(seed=args.seed, target_tile=args.target_tile, depth=3)
        print(json.dumps({key: result[key] for key in ("max_tile", "steps", "score", "target_tile", "target_reached", "policy")}, indent=2))
        return
    trainer = TRMTrainer(TrainerConfig())
    if args.serve:
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("Install uvicorn to use --serve") from exc
        from backend.api import create_app

        uvicorn.run(create_app(trainer), host=args.host, port=args.port)
        return
    if args.episodes:
        result = trainer.run_cycle(episodes=args.episodes, updates=max(1, args.episodes // 2))
    else:
        trainer.start_async()
        try:
            input("Training in the background; press Enter to stop.\n")
        finally:
            trainer.stop()
        result = trainer.status()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
