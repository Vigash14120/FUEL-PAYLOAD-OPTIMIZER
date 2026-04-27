/**
 * MetricsPage — reward curves, profit trends, regression model comparison
 */
import { useState, useEffect } from 'react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { api } from '../api'

const COLORS = {
  reward: '#3b82f6',
  profit: '#10b981',
  loss:   '#f59e0b',
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <div style={{ marginBottom:'4px', color:'var(--text-400)' }}>Episode {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </div>
      ))}
    </div>
  )
}

function smoothSeries(arr, window = 20) {
  return arr.map((v, i) => {
    const slice = arr.slice(Math.max(0, i - window), i + 1)
    return slice.reduce((a, b) => a + b, 0) / slice.length
  })
}

function buildChartData(rewards, profits) {
  if (!rewards?.length) return []
  const smooth_r = smoothSeries(rewards)
  const smooth_p = smoothSeries(profits)
  return rewards.map((_, i) => ({
    episode: i + 1,
    reward:  parseFloat(rewards[i]?.toFixed(3)),
    profit:  parseFloat(profits[i]?.toFixed(0)),
    smooth_reward: parseFloat(smooth_r[i]?.toFixed(3)),
    smooth_profit: parseFloat(smooth_p[i]?.toFixed(0)),
  })).filter((_, i) => i % Math.max(1, Math.floor(rewards.length / 300)) === 0) // downsample
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState(null)

  const fetchMetrics = async () => {
    try { setMetrics(await api.metrics()) } catch (_) {}
  }

  useEffect(() => { fetchMetrics() }, [])

  const dqnData = metrics?.dqn  ? buildChartData(metrics.dqn.rewards,  metrics.dqn.profits)  : []
  const ppoData = metrics?.ppo  ? buildChartData(metrics.ppo.rewards,  metrics.ppo.profits)  : []
  const regMetrics = metrics?.regression?.metrics
  const barData = regMetrics
    ? Object.entries(regMetrics).map(([name, m]) => ({
        name: name === 'random_forest' ? 'Random Forest' : 'SVM',
        MAE:  m.mae,
        R2:   m.r2 * 100,
        CV_MAE: m.cv_mae,
      }))
    : []

  const hasData = dqnData.length || ppoData.length || barData.length

  return (
    <div style={{ display:'grid', gap:'1.5rem' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div>
          <h2 className="section-heading">Training Metrics</h2>
          <p className="section-sub">Reward curves, profit trends, and regression model accuracy.</p>
        </div>
        <button className="btn btn-secondary" onClick={fetchMetrics}>↻ Refresh</button>
      </div>

      {!hasData && (
        <div className="card" style={{ textAlign:'center', padding:'4rem', color:'var(--text-400)' }}>
          <div style={{ fontSize:'3rem', marginBottom:'1rem' }}>📊</div>
          <div style={{ fontWeight:600, marginBottom:'4px' }}>No metrics yet</div>
          <div style={{ fontSize:'0.82rem' }}>Train the regression model and at least one RL agent to see charts.</div>
        </div>
      )}

      {/* Regression Comparison */}
      {barData.length > 0 && (
        <div className="card">
          <div className="card-title">
            <span className="icon" style={{ background:'rgba(59,130,246,0.15)' }}>🔬</span>
            Regression Model Comparison
            <span className="badge badge-blue" style={{ marginLeft:'auto' }}>{metrics.regression.best === 'random_forest' ? 'Random Forest Wins' : 'SVM Wins'}</span>
          </div>
          <div className="grid-2">
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-400)', marginBottom:'8px' }}>MAE — Mean Absolute Error (kg) ↓ lower is better</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={barData} margin={{ top:4, right:16, bottom:0, left:0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="name" tick={{ fill:'var(--text-400)', fontSize:11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'var(--text-400)', fontSize:11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="MAE" fill="var(--primary)" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-400)', marginBottom:'8px' }}>R² Score (%) ↑ higher is better</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={barData} margin={{ top:4, right:16, bottom:0, left:0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="name" tick={{ fill:'var(--text-400)', fontSize:11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'var(--text-400)', fontSize:11 }} axisLine={false} tickLine={false} domain={[0,100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="R2" fill="var(--success)" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metrics table */}
          <div style={{ marginTop:'1rem', overflow:'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th><th>MAE (kg)</th><th>R²</th><th>CV-MAE (kg)</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {barData.map(row => (
                  <tr key={row.name}>
                    <td>{row.name}</td>
                    <td>{row.MAE}</td>
                    <td>{(row.R2 / 100).toFixed(4)}</td>
                    <td>{row.CV_MAE}</td>
                    <td><span className={`badge badge-${row.name.includes(metrics.regression.best === 'random_forest' ? 'Forest' : 'SVM') ? 'green' : 'amber'}`}>
                      {row.name.includes(metrics.regression.best === 'random_forest' ? 'Forest' : 'SVM') ? '⭐ Best' : 'Baseline'}
                    </span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* DQN Curves */}
      {dqnData.length > 0 && (
        <div className="card">
          <div className="card-title">
            <span className="icon" style={{ background:'rgba(59,130,246,0.15)' }}>📈</span>
            DQN Training Curves
            <span className="badge badge-blue" style={{ marginLeft:'auto' }}>
              Final Avg Profit: ${metrics.dqn.final_avg_profit?.toLocaleString()}
            </span>
          </div>
          <div className="grid-2">
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-400)', marginBottom:'8px' }}>Reward per Episode</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={dqnData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                  <defs>
                    <linearGradient id="colorR" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={COLORS.reward} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={COLORS.reward} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="episode" tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="reward" stroke={COLORS.reward} fill="url(#colorR)" strokeWidth={1.5} name="Reward" dot={false} />
                  <Line type="monotone" dataKey="smooth_reward" stroke={COLORS.reward} strokeWidth={2.5} dot={false} name="Smoothed" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-400)', marginBottom:'8px' }}>Profit per Episode (USD)</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={dqnData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                  <defs>
                    <linearGradient id="colorP" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={COLORS.profit} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={COLORS.profit} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="episode" tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="profit" stroke={COLORS.profit} fill="url(#colorP)" strokeWidth={1.5} name="Profit" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* PPO Curves */}
      {ppoData.length > 0 && (
        <div className="card">
          <div className="card-title">
            <span className="icon" style={{ background:'rgba(139,92,246,0.15)' }}>📈</span>
            PPO Training Curves
            <span className="badge badge-purple" style={{ marginLeft:'auto' }}>
              Final Avg Profit: ${metrics.ppo.final_avg_profit?.toLocaleString()}
            </span>
          </div>
          <div className="grid-2">
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-400)', marginBottom:'8px' }}>Reward per Episode</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={ppoData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                  <defs>
                    <linearGradient id="colorPPOR" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#8b5cf6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="episode" tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="reward" stroke="#8b5cf6" fill="url(#colorPPOR)" strokeWidth={1.5} name="Reward" dot={false} />
                  <Line type="monotone" dataKey="smooth_reward" stroke="#8b5cf6" strokeWidth={2.5} dot={false} name="Smoothed" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-400)', marginBottom:'8px' }}>Profit per Episode (USD)</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={ppoData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                  <defs>
                    <linearGradient id="colorPPOP" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={COLORS.profit} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={COLORS.profit} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="episode" tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'var(--text-400)', fontSize:10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="profit" stroke={COLORS.profit} fill="url(#colorPPOP)" strokeWidth={1.5} name="Profit" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
