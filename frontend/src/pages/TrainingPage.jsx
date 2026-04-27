/**
 * TrainingPage — controls to train regression & RL agents
 * Receives live updates via WebSocket
 */
import { useState, useEffect, useRef } from 'react'
import { api } from '../api'

function LogLine({ entry }) {
  return (
    <div className={`log-line ${entry.type}`}>
      <span className="log-time">{entry.time}</span>
      <span className="log-msg">{entry.msg}</span>
    </div>
  )
}

export default function TrainingPage({ status, onStatusUpdate, wsMessages }) {
  const [rSamples,   setRSamples]   = useState(10000)
  const [rlAgent,    setRlAgent]    = useState('DQN')
  const [rlEpisodes, setRlEpisodes] = useState(500)
  const [logs,       setLogs]       = useState([])
  const [progress,   setProgress]   = useState(null)  // { episode, total, avg_reward, avg_profit }
  const logRef = useRef(null)

  const addLog = (msg, type = 'info') => {
    const time = new Date().toLocaleTimeString()
    setLogs(prev => [...prev.slice(-100), { time, msg, type }])
  }

  useEffect(() => {
    if (!wsMessages) return
    const msg = wsMessages
    if (msg.type === 'regression_done') {
      addLog(`✓ Regression training complete. Best: ${msg.data.best}`, 'success')
      onStatusUpdate()
    } else if (msg.type === 'progress') {
      const d = msg.data
      setProgress(d)
      if (d.episode % 100 === 0 || d.episode === d.total) {
        addLog(
          `${d.agent} ep ${d.episode}/${d.total} | avg_reward=${d.avg_reward} | avg_profit=$${d.avg_profit}`,
          'info'
        )
      }
    } else if (msg.type === 'rl_done') {
      addLog(`✓ ${msg.data.agent} training complete! Avg profit: $${msg.data.final_avg_profit}`, 'success')
      setProgress(null)
      onStatusUpdate()
    } else if (msg.type === 'error') {
      addLog(`✗ Error: ${msg.detail}`, 'error')
      setProgress(null)
    }
  }, [wsMessages])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const handleTrainRegression = async () => {
    addLog(`Starting regression training (${rSamples.toLocaleString()} samples)…`)
    await api.trainRegression(rSamples)
  }

  const handleTrainRL = async () => {
    if (!status?.regression_trained) {
      addLog('Train the regression model first.', 'warn')
      return
    }
    addLog(`Starting ${rlAgent} training (${rlEpisodes} episodes)…`)
    await api.trainRL(rlAgent, rlEpisodes)
  }

  const progressPct = progress
    ? Math.round((progress.episode / progress.total) * 100)
    : 0

  return (
    <div style={{ display:'grid', gap:'1.5rem' }}>
      <div>
        <h2 className="section-heading">Training Pipeline</h2>
        <p className="section-sub">Train the regression model, then the RL agent using the simulation environment.</p>
      </div>

      {/* Step 1 — Regression */}
      <div className="card">
        <div className="card-title">
          <span className="icon" style={{ background:'rgba(59,130,246,0.15)' }}>🔬</span>
          Step 1 — Regression Model (SVM / Random Forest)
          {status?.regression_trained && <span className="badge badge-green" style={{ marginLeft:'auto' }}>Trained ✓</span>}
        </div>
        <p style={{ color:'var(--text-400)', fontSize:'0.82rem', marginBottom:'1.5rem', lineHeight:1.6 }}>
          Generates synthetic flight log data and trains both SVM and Random Forest regressors
          to predict fuel burn (kg) given passenger count, weather, fuel price, distance, and cargo weight.
          The best model is persisted and used by the RL simulation environment.
        </p>

        <div style={{ display:'grid', gridTemplateColumns:'1fr auto', gap:'1rem', alignItems:'end' }}>
          <div className="form-group">
            <label className="form-label">Training Samples</label>
            <div className="slider-row">
              <input
                type="range" className="form-slider"
                min={1000} max={50000} step={1000}
                value={rSamples}
                onChange={e => setRSamples(Number(e.target.value))}
              />
              <span className="slider-value">{rSamples.toLocaleString()}</span>
            </div>
          </div>
          <button
            id="btn-train-regression"
            className="btn btn-primary"
            onClick={handleTrainRegression}
            disabled={status?.training_running}
          >
            {status?.training_running ? <span className="spinner"/> : '🚀'}
            Train Regression
          </button>
        </div>
      </div>

      {/* Step 2 — RL Agent */}
      <div className="card">
        <div className="card-title">
          <span className="icon" style={{ background:'rgba(139,92,246,0.15)' }}>🧠</span>
          Step 2 — Deep RL Agent (DQN / PPO)
          {(status?.dqn_trained || status?.ppo_trained) && (
            <span className="badge badge-purple" style={{ marginLeft:'auto' }}>
              {status.dqn_trained && status.ppo_trained ? 'Both trained ✓' :
               status.dqn_trained ? 'DQN trained ✓' : 'PPO trained ✓'}
            </span>
          )}
        </div>
        <p style={{ color:'var(--text-400)', fontSize:'0.82rem', marginBottom:'1.5rem', lineHeight:1.6 }}>
          Trains the selected RL agent against the simulation environment.
          <strong style={{ color:'var(--text-300)' }}> DQN</strong> uses a discretized action grid with ε-greedy exploration and experience replay.
          <strong style={{ color:'var(--text-300)' }}> PPO</strong> operates in the full continuous action space with GAE advantage estimation.
        </p>

        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr auto', gap:'1rem', alignItems:'end' }}>
          <div className="form-group">
            <label className="form-label">Agent</label>
            <select className="form-select" value={rlAgent} onChange={e => setRlAgent(e.target.value)}>
              <option value="DQN">DQN — Deep Q-Network (discrete)</option>
              <option value="PPO">PPO — Proximal Policy Opt. (continuous)</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Episodes</label>
            <div className="slider-row">
              <input
                type="range" className="form-slider"
                min={50} max={2000} step={50}
                value={rlEpisodes}
                onChange={e => setRlEpisodes(Number(e.target.value))}
              />
              <span className="slider-value">{rlEpisodes}</span>
            </div>
          </div>
          <button
            id="btn-train-rl"
            className="btn btn-accent"
            onClick={handleTrainRL}
            disabled={status?.training_running || !status?.regression_trained}
          >
            {status?.training_running ? <span className="spinner"/> : '🎯'}
            Train {rlAgent}
          </button>
        </div>
      </div>

      {/* Progress */}
      {progress && (
        <div className="card">
          <div className="card-title">⏳ Training Progress — {progress.agent}</div>
          <div style={{ marginBottom:'8px', display:'flex', justifyContent:'space-between', fontSize:'0.8rem', color:'var(--text-300)' }}>
            <span>Episode {progress.episode} / {progress.total}</span>
            <span>{progressPct}%</span>
          </div>
          <div className="progress-bar-wrap">
            <div className="progress-bar-fill" style={{ width:`${progressPct}%` }} />
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'1rem', marginTop:'1rem' }}>
            <div>
              <div style={{ fontSize:'0.72rem', color:'var(--text-400)', marginBottom:'4px' }}>AVG REWARD</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.9rem', color:'var(--primary)' }}>{progress.avg_reward}</div>
            </div>
            <div>
              <div style={{ fontSize:'0.72rem', color:'var(--text-400)', marginBottom:'4px' }}>AVG PROFIT</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.9rem', color:'var(--success)' }}>${progress.avg_profit?.toLocaleString()}</div>
            </div>
            <div>
              <div style={{ fontSize:'0.72rem', color:'var(--text-400)', marginBottom:'4px' }}>LOSS</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.9rem', color:'var(--warning)' }}>{progress.loss ?? '—'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Log Console */}
      <div className="card">
        <div className="card-title" style={{ marginBottom:'0.75rem' }}>📋 Training Log</div>
        <div className="log-console" ref={logRef}>
          {logs.length === 0
            ? <span style={{ color:'var(--text-400)' }}>Waiting for training events…</span>
            : logs.map((l, i) => <LogLine key={i} entry={l} />)
          }
        </div>
      </div>
    </div>
  )
}
