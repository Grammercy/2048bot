# 2048bot

2048bot trains a 4×4 2048 agent with self-play. The main learner is a small
recurrent policy/value network. It reuses one latent transition for several
reasoning steps, then updates from discounted episode returns. The reward puts
most of its weight on surviving moves and empty-board space, so the agent is
trained to keep playing long enough to build larger tiles.

## Run it

Install the Python and frontend dependencies:

```sh
python -m pip install -r requirements.txt
npm install
```

Start the API and dashboard in separate terminals:

```sh
python -m backend.server --port 8000
npm run dev
```

Open <http://localhost:5173>. The dashboard shows an idle board until the API
returns a real episode, then polls live training metrics and the trainer's
recorded move history. The start/stop control calls the training API directly;
the replay, chart, episode table, telemetry, and board-position summary all
render API data (or an explicit empty/offline state). A fresh server labels its
deterministic warm-start game as `heuristic-demo`; a loaded checkpoint produces
a `trm` replay.

To run a bounded training cycle without the UI:

```sh
python train.py --episodes 100
```

For a target-seeking evaluation run, use the cached expectimax planner.  The
checked-in deterministic seed reaches the 4096 tile and stops as soon as the
target is formed:

```sh
python train.py --evaluate-strong --seed 2059 --target-tile 4096
```

The model checkpoint and metrics are written under `data/` after training.

## API

The standard-library server exposes these endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/train/stats` | Live episodes, mean highest tile, lifespan, losses, entropy, and replay size |
| POST | `/api/train` | Start background training; body can include `{ "games": 100 }` |
| POST | `/api/train/stop` | Stop background training |
| GET | `/api/replay?source=training` | Latest self-play game and move history |
| GET | `/api/state` | Interactive board state |
| POST | `/api/move` | Move the interactive board; body `{ "action": "left" }` |
| POST | `/api/new` | Reset the interactive board |
| GET | `/api/model` | TRM configuration and optimizer step |

`backend.api` also exposes the same controls through FastAPI/Uvicorn when a
deployment needs an ASGI server.  `POST /api/games/play` accepts
`{"policy":"expectimax"}` for the target-seeking evaluator; the default
`trm` policy remains available for model comparisons.  The standard-library
server accepts the same policy on `/api/evaluate` (for example,
`/api/evaluate?games=1&policy=expectimax`).

## Checks

```sh
python -m unittest discover -s tests -v
npm run build
```
