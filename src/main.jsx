import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const emptyGrid = Array.from({ length: 4 }, () => Array(4).fill(0))

const directionGlyph = { left: '←', up: '↑', right: '→', down: '↓' }
const actionNames = ['up', 'right', 'down', 'left']

const tileClass = (value) => value ? `tile tile-${value}` : 'tile tile-empty'

function LogoMark() {
  return <div className="logo-mark" aria-hidden="true"><span>2</span><span>0</span><span>4</span><span>8</span></div>
}

function Tile({ value }) {
  return <div className={tileClass(value)}>{value ? <span>{value}</span> : null}</div>
}

function flatGrid(grid) {
  return grid.flatMap((row, rowIndex) => row.map((value, colIndex) => ({ value, row: rowIndex, col: colIndex })))
}

function highestTile(grid) {
  return grid.flat().reduce((highest, value) => Math.max(highest, Number(value) || 0), 0)
}

function normalizeBoard(board) {
  if (!Array.isArray(board)) return null
  if (board.length === 4 && board.every((row) => Array.isArray(row) && row.length === 4)) return board.map((row) => row.map((value) => Number(value) || 0))
  if (board.length === 16 && board.every((value) => Number.isFinite(Number(value)))) return [0, 1, 2, 3].map((row) => board.slice(row * 4, row * 4 + 4).map((value) => Number(value) || 0))
  return null
}

function normalizeAction(turn) {
  const named = turn?.action_name ?? turn?.action
  if (typeof named === 'string' && directionGlyph[named.toLowerCase()]) return named.toLowerCase()
  const numeric = Number(turn?.action)
  return Number.isInteger(numeric) && actionNames[numeric] ? actionNames[numeric] : ''
}

function normalizeReplay(payload) {
  const record = payload?.replay || payload?.state || payload
  const history = Array.isArray(record?.history) ? record.history : Array.isArray(record?.moves) ? record.moves : []
  const firstBoard = normalizeBoard(record?.initial_board) || normalizeBoard(history[0]?.board_before) || normalizeBoard(record?.board) || emptyGrid
  const frames = [firstBoard]
  const moves = []
  const scores = [Number(record?.initial_score || 0)]
  history.forEach((turn) => {
    const next = normalizeBoard(turn.board_after) || normalizeBoard(turn.board)
    if (!next) return
    frames.push(next)
    moves.push(normalizeAction(turn))
    scores.push(Number(turn.score || 0))
  })
  return {
    frames,
    moves,
    scores,
    score: Number(record?.score || scores[scores.length - 1] || 0),
    hasHistory: frames.length > 1,
    episode: record?.episode == null ? null : Number(record.episode),
    policy: record?.policy || record?.algorithm || null,
  }
}

function staticTiles(grid) {
  return flatGrid(grid)
    .filter((cell) => cell.value)
    .map((cell, index) => ({ ...cell, fromRow: cell.row, fromCol: cell.col, key: `static-${index}-${cell.row}-${cell.col}` }))
}

function motionTiles(previous, next, action) {
  const motions = []
  const coveredTargets = new Set()
  const direction = directionGlyph[action] ? action : null

  if (direction) {
    for (let lineIndex = 0; lineIndex < 4; lineIndex += 1) {
      const axis = [0, 1, 2, 3]
      const ordered = direction === 'right' || direction === 'down' ? axis.reverse() : axis
      const positions = direction === 'left' || direction === 'right'
        ? ordered.map((col) => [lineIndex, col])
        : ordered.map((row) => [row, lineIndex])
      const packed = positions.map(([row, col]) => ({ value: previous[row][col], row, col })).filter((cell) => cell.value)
      let sourceIndex = 0
      let targetIndex = 0
      while (sourceIndex < packed.length) {
        const target = positions[targetIndex]
        const first = packed[sourceIndex]
        const second = packed[sourceIndex + 1]
        const merges = second && second.value === first.value
        const sources = merges ? [first, second] : [first]
        sources.forEach((source, mergeIndex) => {
          motions.push({ value: source.value, row: target[0], col: target[1], fromRow: source.row, fromCol: source.col, key: `line-${lineIndex}-${sourceIndex}-${mergeIndex}` })
        })
        coveredTargets.add(`${target[0]}-${target[1]}`)
        sourceIndex += merges ? 2 : 1
        targetIndex += 1
      }
    }

    flatGrid(next).filter((cell) => cell.value && !coveredTargets.has(`${cell.row}-${cell.col}`)).forEach((cell, index) => {
      motions.push({ ...cell, fromRow: cell.row, fromCol: cell.col, key: `spawn-${index}`, spawn: true })
    })
    return motions
  }

  // Initial loads or malformed records have no action direction. Keep these
  // tiles stationary and mark new values as spawns without inventing motion.
  return flatGrid(next).filter((cell) => cell.value).map((cell, index) => ({
    ...cell,
    fromRow: cell.row,
    fromCol: cell.col,
    key: `static-${index}-${cell.row}-${cell.col}`,
    spawn: !previous[cell.row]?.[cell.col],
  }))
}

function Board({ grid, move }) {
  const previousGrid = React.useRef(grid)
  const [motion, setMotion] = React.useState({ tiles: staticTiles(grid), active: true })

  useEffect(() => {
    if (JSON.stringify(previousGrid.current) === JSON.stringify(grid)) return undefined
    const nextMotion = motionTiles(previousGrid.current, grid, move)
    previousGrid.current = grid
    setMotion({ tiles: nextMotion, active: false })
    const frame = requestAnimationFrame(() => requestAnimationFrame(() => setMotion({ tiles: nextMotion, active: true })))
    const settle = window.setTimeout(() => setMotion({ tiles: staticTiles(grid), active: true }), 290)
    return () => {
      cancelAnimationFrame(frame)
      window.clearTimeout(settle)
    }
  }, [grid, move])

  return <div className="board" role="grid" aria-label="Current 2048 board">
    <div className="board-grid" aria-hidden="true">
      {Array.from({ length: 16 }, (_, index) => <Tile value={0} key={`slot-${index}`} />)}
    </div>
    <div className="moving-layer" aria-live="polite">
      {motion.tiles.map((tile) => {
        const row = motion.active ? tile.row : tile.fromRow
        const col = motion.active ? tile.col : tile.fromCol
        return <div className={`${tileClass(tile.value)} moving-tile cell-x${col} cell-y${row} ${tile.spawn ? 'spawn-tile' : ''}`} key={tile.key}><span>{tile.value}</span></div>
      })}
    </div>
  </div>
}

function MetricCard({ label, value, suffix, detail, tone = 'normal', icon }) {
  return <article className={`metric-card metric-${tone}`}>
    <div className="metric-head"><span>{label}</span><span className="metric-icon">{icon}</span></div>
    <div className="metric-value">{value}<small>{suffix}</small></div>
    <div className="metric-detail">{detail}</div>
  </article>
}

function Sparkline({ values }) {
  if (!values.length) return <div className="chart-empty">waiting for recorded episodes</div>
  const max = Math.max(...values)
  const min = Math.min(...values)
  const range = max - min || 1
  const points = values.map((value, i) => {
    const x = values.length === 1 ? 50 : (i / (values.length - 1)) * 100
    const y = 92 - ((value - min) / range) * 73
    return `${x},${y}`
  }).join(' ')
  const area = `0,100 ${points} 100,100`
  return <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Highest tile by recorded episode" role="img">
    <polygon points={area} fill="#ef7a3a" fillOpacity=".10" />
    <polyline points={points} fill="none" stroke="#f2b35a" strokeWidth="1.7" vectorEffect="non-scaling-stroke" />
  </svg>
}

function Heatmap({ frames }) {
  const counts = Array(16).fill(0)
  frames.forEach((grid) => grid.flat().forEach((value, index) => { if (value) counts[index] += 1 }))
  const maximum = Math.max(...counts, 0)
  const cells = counts.map((count) => maximum ? Math.max(1, Math.round((count / maximum) * 7)) : 0)
  return <div className="heatmap" aria-label="Recorded tile occupancy map">
    {cells.map((intensity, i) => <span key={i} className={`heat-${intensity}`} />)}
  </div>
}

function ChartCard({ snapshot }) {
  const values = Array.isArray(snapshot?.recent_max_tiles) ? snapshot.recent_max_tiles.map(Number).filter(Number.isFinite) : []
  const liveAverage = Number(snapshot?.mean_max_tile ?? snapshot?.average_highest_tile)
  const hasAverage = Number.isFinite(liveAverage) && Number(snapshot?.episodes || 0) > 0
  const latest = values.at(-1)
  const first = values[0]
  const trend = values.length > 1 && first > 0 ? ((latest - first) / first) * 100 : null
  const maxValue = Math.max(...values, 0)
  const yLabels = maxValue ? [maxValue, maxValue / 2, maxValue / 4, maxValue / 8, 0] : [0, 0, 0, 0, 0]
  const firstEpisode = Math.max(1, Number(snapshot?.episodes || values.length) - values.length + 1)
  const lastEpisode = Number(snapshot?.episodes || values.length)
  return <section className="panel chart-panel">
    <div className="panel-heading">
      <div><div className="eyebrow">Trajectory</div><h2>Highest tile by episode</h2></div>
      <div className="chart-stat"><strong>{hasAverage ? Math.round(liveAverage).toLocaleString() : '—'}</strong><span className={trend == null ? '' : trend >= 0 ? 'positive' : 'negative'}>{trend == null ? 'no trend yet' : `${trend >= 0 ? '↑' : '↓'} ${Math.abs(trend).toFixed(1)}% vs first`}</span><small>{values.length ? `last ${values.length} recorded` : 'no recorded episodes'}</small></div>
    </div>
    <div className="chart-wrap">
      <div className="chart-y-labels">{yLabels.map((value, index) => <span key={index}>{Math.round(value).toLocaleString()}</span>)}</div>
      <div className="chart-main">
        <div className="chart-grid"><span /><span /><span /><span /><span /></div>
        <Sparkline values={values} />
        <div className="chart-x-labels"><span>ep {firstEpisode}</span><span>ep {lastEpisode}</span></div>
      </div>
    </div>
    <div className="chart-foot"><span><i className="legend-dot orange" /> one point / episode</span><span className="chart-foot-note">{values.length ? `window ${firstEpisode}–${lastEpisode}` : 'waiting for API data'}</span></div>
  </section>
}

function EpisodeList({ snapshot }) {
  const runs = Array.isArray(snapshot?.recent_games) ? snapshot.recent_games.slice(-4).reverse() : []
  return <section className="panel episodes-panel">
    <div className="panel-heading compact"><div><div className="eyebrow">Rollout queue</div><h2>Recent episodes</h2></div><span className="hint">{runs.length ? `${runs.length} shown` : 'no data'}</span></div>
    <div className="episode-list">
      {runs.length ? runs.map((game, index) => {
        const tile = Number(game.highest_tile || 0)
        const episode = Number(game.episode || 0)
        return <div className="episode-row" key={`${episode}-${index}`}>
          <div className="episode-id">{episode ? `#${String(episode).padStart(4, '0')}` : 'episode —'}</div><div className={`mini-tile ${tile > 4 ? 'mini-tile-high' : 'mini-tile-low'}`}>{tile ? tile.toLocaleString() : '—'}</div><div className="episode-score"><strong>{Number(game.score || 0).toLocaleString()}</strong><small>score</small></div><div className="episode-time">{game.moves == null ? '—' : `${Number(game.moves).toLocaleString()} moves`}</div><div className="episode-when">recorded</div>
        </div>
      }) : <div className="empty-state">No episodes recorded yet.</div>}
    </div>
  </section>
}

function Telemetry({ frame, snapshot }) {
  const trained = Boolean(snapshot && !snapshot.demo_mode)
  const metric = (value, digits = 3) => trained && value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—'
  const rows = [
    ['policy entropy', metric(snapshot?.last_entropy), trained ? 'recorded' : 'no update', 'steady'],
    ['value loss', metric(snapshot?.last_value_loss), trained ? 'recorded' : 'no update', 'steady'],
    ['exploration', trained && snapshot?.epsilon != null ? `${(Number(snapshot.epsilon) * 100).toFixed(1)}%` : '—', trained ? 'schedule' : 'no update', 'steady'],
    ['replay buffer', trained && snapshot?.replay_size != null ? Number(snapshot.replay_size).toLocaleString() : '—', trained ? 'samples' : 'no update', 'up'],
  ]
  const thinkingSteps = Number(snapshot?.reasoning_steps || 0)
  const status = snapshot ? (snapshot.running ? 'live' : snapshot.demo_mode ? 'demo' : 'idle') : 'offline'
  return <section className="panel telemetry-panel">
    <div className="panel-heading compact"><div><div className="eyebrow">Latent loop</div><h2>Training telemetry</h2></div><span className="pulse-label"><i /> {status}</span></div>
    <div className="telemetry-rows">{rows.map(([name, value, change, trend]) => <div className="telemetry-row" key={name}><span>{name}</span><strong>{value}</strong><small className={trend}>{change}</small></div>)}</div>
    <div className="latent-strip"><span>thinking steps</span><strong>{thinkingSteps || '—'}</strong><span className="strip-bars">{Array.from({ length: thinkingSteps }).map((_, i) => <i key={i} className={i <= (thinkingSteps ? frame % thinkingSteps : -1) ? 'filled' : ''} />)}</span></div>
  </section>
}

function ReplayCard({ grid, frame, replayMoves, replayScore, replayEpisode, replayPolicy, hasHistory, playing, setPlaying, onStep, onReset, speed, setSpeed }) {
  const move = hasHistory ? (frame > 0 ? replayMoves[frame - 1] || 'unknown' : 'start') : 'waiting'
  const agent = replayPolicy === 'heuristic-demo' ? 'heuristic demo' : replayPolicy === 'trm' ? 'TRM self-play' : replayPolicy === 'expectimax' ? 'expectimax target run' : replayPolicy || 'awaiting API'
  return <section className="panel replay-panel">
    <div className="panel-heading compact"><div><div className="eyebrow">Recorded replay</div><h2>Watch a game</h2></div><div className="agent-pill"><span />{agent}</div></div>
    <Board grid={grid} move={move} />
    <div className="replay-meta"><span>episode <strong>{replayEpisode || '—'}</strong></span><span>move <strong>{String(frame).padStart(2, '0')}</strong> / {replayMoves.length}</span><span>score <strong>{replayScore.toLocaleString()}</strong></span></div>
    <div className="replay-controls"><button className="control-button" onClick={onReset} aria-label="Reset replay" disabled={!hasHistory}>↺</button><button className="play-button" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause replay' : 'Play replay'} disabled={!hasHistory}>{playing ? 'Ⅱ' : '▶'}</button><button className="control-button" onClick={onStep} aria-label="Next replay frame" disabled={!hasHistory}>→</button><label className="speed-control"><span>speed</span><select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} disabled={!hasHistory}><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label></div>
    <div className="move-line"><span className="move-chip">{directionGlyph[move] || '·'}</span><span>{move === 'waiting' ? 'waiting for a recorded episode' : move === 'start' ? 'initial board' : <><span>recorded action </span><strong>{move}</strong></>}</span><span className="confidence">{move === 'waiting' ? 'offline' : 'from replay'}</span></div>
  </section>
}

function TargetCard({ run, busy, onRun }) {
  const reached = Boolean(run?.target_reached)
  const maxTile = Number(run?.max_tile || 0)
  const steps = Number(run?.steps || 0)
  return <section className={`target-card ${reached ? 'target-card-reached' : ''}`}>
    <div className="target-orbit" aria-hidden="true"><span /><span /><span /></div>
    <div className="target-copy">
      <div className="eyebrow">Target protocol</div>
      <h2>{reached ? '4096 is on the board.' : 'Push the board to 4096.'}</h2>
      <p>{reached ? `Expectimax found the tile in ${steps.toLocaleString()} moves.` : 'Run the deep evaluator and watch the board build a stable corner.'}</p>
      <div className="target-meta"><span className="target-chip"><i /> expectimax / depth 3</span><span className="target-seed">seed 2059</span></div>
    </div>
    <div className="target-result">
      <span className="target-result-label">highest tile</span>
      <strong>{maxTile ? maxTile.toLocaleString() : '—'}</strong>
      <button className="target-button" type="button" onClick={onRun} disabled={busy}>{busy ? 'running…' : reached ? 'run again' : 'run evaluator'} <span>↗</span></button>
    </div>
  </section>
}

function App() {
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [trainerSnapshot, setTrainerSnapshot] = useState(null)
  const [remoteReplay, setRemoteReplay] = useState(null)
  const [actionBusy, setActionBusy] = useState(false)
  const [plannerBusy, setPlannerBusy] = useState(false)
  const [plannerRun, setPlannerRun] = useState(null)
  const replayData = useMemo(() => normalizeReplay(remoteReplay), [remoteReplay])
  const replayFrames = replayData.frames
  const replayMoves = replayData.moves
  const grid = replayFrames[Math.min(frame, replayFrames.length - 1)]

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      let received = false
      for (const path of ['/api/train/stats', '/api/status']) {
        try {
          const response = await fetch(path, { headers: { Accept: 'application/json' } })
          if (!response.ok) continue
          const payload = await response.json()
          if (!cancelled) {
            setTrainerSnapshot(payload)
            received = true
          }
          break
        } catch { /* API may be offline */ }
      }
      if (!received && !cancelled) setTrainerSnapshot(null)
    }
    poll()
    const timer = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(timer) }
  }, [])

  useEffect(() => {
    let cancelled = false
    const pollReplay = async () => {
      for (const path of ['/api/replay?source=training', '/api/replay']) {
        try {
          const response = await fetch(path, { headers: { Accept: 'application/json' } })
          if (!response.ok) continue
          const payload = await response.json()
          const parsed = normalizeReplay(payload)
          if (!cancelled && parsed.frames.length > 1) setRemoteReplay(payload)
          break
        } catch { /* API may be offline */ }
      }
    }
    pollReplay()
    const timer = setInterval(pollReplay, 2500)
    return () => { cancelled = true; clearInterval(timer) }
  }, [])

  const snapshot = trainerSnapshot
  const recordedEpisodes = Number(snapshot?.games ?? snapshot?.episodes ?? 0)
  const averageValue = Number(snapshot?.average_highest_tile ?? snapshot?.mean_max_tile)
  const averageTile = Number.isFinite(averageValue) && averageValue > 0 ? Math.round(averageValue).toLocaleString() : '—'
  const replayHighestTile = highestTile(grid)
  const reportedBestTile = Number(snapshot?.best_tile || 0)
  const bestTile = (reportedBestTile || (replayData.hasHistory ? replayHighestTile : 0)) ? (reportedBestTile || replayHighestTile).toLocaleString() : '—'
  const games = recordedEpisodes.toLocaleString()
  const episodeRateValue = Number(snapshot?.episodes_per_second)
  const episodeRate = Number.isFinite(episodeRateValue) && episodeRateValue > 0 ? `${episodeRateValue.toFixed(1)} ep/s` : '—'
  const averageMovesValue = Number(snapshot?.average_moves ?? snapshot?.mean_episode_length)
  const averageMoves = Number.isFinite(averageMovesValue) && averageMovesValue > 0 ? Math.round(averageMovesValue).toLocaleString() : '—'
  const trainingStatus = snapshot ? (snapshot.running ? 'training' : snapshot.demo_mode ? 'demo data' : 'idle') : 'API offline'
  const statusTime = snapshot?.updated_at ? new Date(Number(snapshot.updated_at) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'

  const analysisFrames = replayData.hasHistory ? replayFrames : []
  const cornerRate = analysisFrames.length ? Math.round(analysisFrames.filter((board) => {
    const max = highestTile(board)
    return max > 0 && [board[0][0], board[0][3], board[3][0], board[3][3]].includes(max)
  }).length / analysisFrames.length * 100) : null

  const toggleTraining = async () => {
    if (actionBusy) return
    setActionBusy(true)
    try {
      const endpoint = snapshot?.running ? '/api/train/stop' : '/api/train'
      const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: '{}' })
      if (response.ok) {
        const payload = await response.json()
        const next = payload.state || payload
        if (next && typeof next === 'object') setTrainerSnapshot(next)
      }
    } catch { /* keep the UI truthful; the next poll will show offline */ }
    finally { setActionBusy(false) }
  }

  const runStrongEvaluation = async () => {
    if (plannerBusy) return
    setPlannerBusy(true)
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ games: 1, policy: 'expectimax' }),
      })
      if (!response.ok) throw new Error(`evaluator returned ${response.status}`)
      const payload = await response.json()
      const result = payload?.episodes?.[0]
      if (result) {
        setPlannerRun(result)
        setRemoteReplay({ replay: result })
        setFrame(0)
        setPlaying(true)
      }
    } catch (error) {
      setPlannerRun({ error: error instanceof Error ? error.message : 'evaluator unavailable' })
    } finally {
      setPlannerBusy(false)
    }
  }

  useEffect(() => {
    setFrame((current) => Math.min(current, replayFrames.length - 1))
  }, [replayFrames.length])

  useEffect(() => {
    if (!playing) return undefined
    if (replayFrames.length < 2) return undefined
    const timer = setInterval(() => setFrame((current) => (current + 1) % replayFrames.length), 1300 / speed)
    return () => clearInterval(timer)
  }, [playing, speed, replayFrames.length])

  return <div className="app-shell">
    <div className="grain" />
    <header className="topbar">
      <div className="brand"><LogoMark /><div><strong>latent play</strong><span>2048 research lab</span></div></div>
      <div className="topbar-actions"><span className="connection"><i />{trainingStatus}</span><button className="train-toggle" onClick={toggleTraining} disabled={actionBusy || !snapshot}>{actionBusy ? 'updating…' : snapshot?.running ? 'stop training' : 'start training'}</button></div>
    </header>

    <main className="content">
      <section className="page-intro">
        <div>
          <div className="kicker"><span className="signal" /> live board intelligence <span className="slash">/</span> 2048</div>
          <h1>Build the corner.<br /><em>Break the ceiling.</em></h1>
          <p>The training room for a tiny agent with one very large ambition: a clean run to the 4096 tile.</p>
        </div>
        <div className="run-actions">
          <span className={`run-status ${plannerRun?.target_reached ? 'run-status-success' : ''}`}><i />{plannerRun?.target_reached ? 'target reached' : trainingStatus}</span>
          <button className="primary-button" type="button" onClick={runStrongEvaluation} disabled={plannerBusy}>{plannerBusy ? 'running evaluator…' : 'run 4096 evaluator'} <span>↗</span></button>
        </div>
      </section>
      <TargetCard run={plannerRun} busy={plannerBusy} onRun={runStrongEvaluation} />
      <section className="metrics-grid">
        <MetricCard label="highest tile" value={bestTile} detail={reportedBestTile ? (snapshot?.demo_mode ? 'highest tile in startup demo' : 'best tile seen in self-play') : 'waiting for a recorded game'} tone="orange" icon="↗" />
        <MetricCard label="avg. highest" value={averageTile} detail={snapshot?.demo_mode ? 'startup demo mean' : recordedEpisodes ? 'mean of recorded episodes' : 'waiting for training data'} tone="green" icon="⌁" />
        <MetricCard label="mean lifespan" value={averageMoves} suffix={averageMoves === '—' ? '' : ' moves'} detail={recordedEpisodes ? 'mean recorded episode length' : 'waiting for training data'} tone="blue" icon="◌" />
        <MetricCard label="episodes trained" value={games} suffix=" episodes" detail={`last update ${statusTime} · ${episodeRate}`} tone="purple" icon="⁙" />
      </section>

      <div className="workspace-grid">
        <ReplayCard grid={grid} replayMoves={replayMoves} replayScore={replayData.scores[Math.min(frame, replayData.scores.length - 1)] || 0} replayEpisode={replayData.episode} replayPolicy={replayData.policy} hasHistory={replayData.hasHistory} frame={frame} playing={playing} setPlaying={setPlaying} onStep={() => setFrame((current) => (current + 1) % replayFrames.length)} onReset={() => { setFrame(0); setPlaying(false) }} speed={speed} setSpeed={setSpeed} />
        <ChartCard snapshot={snapshot} />
        <Telemetry frame={frame} snapshot={snapshot} />
        <section className="panel board-insight-panel"><div className="panel-heading compact"><div><div className="eyebrow">Replay analysis</div><h2>Where tiles land</h2></div><span className="hint">{analysisFrames.length ? `${analysisFrames.length} frames` : 'no data'}</span></div><div className="insight-body"><Heatmap frames={analysisFrames} /><div className="insight-copy"><strong>{cornerRate == null ? 'No replay telemetry' : `Largest tile in a corner: ${cornerRate}%`}</strong><p>{cornerRate == null ? 'A recorded game is required for board-position analysis.' : `Observed across ${analysisFrames.length} frames from the displayed ${replayData.policy || 'recorded'} game.`}</p><div className="insight-tags"><span>{analysisFrames.length ? `${analysisFrames.length} observed` : 'waiting'}</span><span>4 × 4 board</span></div></div></div></section>
        <EpisodeList snapshot={snapshot} />
      </div>
    </main>
    <footer className="statusbar"><span><i className="live-dot" /> {trainingStatus}</span><span>model step <strong>{snapshot?.model_step != null ? Number(snapshot.model_step).toLocaleString() : '—'}</strong></span><span>last sync <strong>{statusTime}</strong></span><span className="footer-right">{snapshot?.algorithm || 'waiting for trainer API'}</span></footer>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
