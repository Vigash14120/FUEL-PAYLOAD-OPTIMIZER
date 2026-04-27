/**
 * InferencePage — single-flight scenario inference with trained agent
 */
import { useState } from 'react'
import { api } from '../api'

function ResultCard({ label, value, sub, color }) {
  return (
    <div className="stat-card" style={{ borderColor: color ? `${color}40` : 'var(--border)' }}>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || 'var(--text-100)' }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

const WEATHER_LABELS = ['Clear', 'Light Clouds', 'Overcast', 'Rain', 'Thunderstorm', 'Severe Storm']

export default function InferencePage({ status }) {
  const [form, setForm] = useState({
    agent:      'DQN',
    pax:        150,
    weather:    0.2,
    fuel_price: 0.85,
    distance:   2500,
  })
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  const weatherLabel = WEATHER_LABELS[Math.floor(form.weather * (WEATHER_LABELS.length - 0.01))]

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  const handleInfer = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.infer(form)
      if (res.status === 'error') setError(res.detail)
      else setResult(res.result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const canInfer = status?.dqn_trained || status?.ppo_trained

  return (
    <div style={{ display:'grid', gap:'1.5rem' }}>
      <div>
        <h2 className="section-heading">Flight Inference</h2>
        <p className="section-sub">Configure a flight scenario and get AI-recommended fuel uplift and cargo allocation.</p>
      </div>

      <div className="grid-2" style={{ gap:'1.5rem' }}>
        {/* Input Form */}
        <div className="card">
          <div className="card-title">
            <span className="icon" style={{ background:'rgba(59,130,246,0.15)' }}>✈️</span>
            Flight Parameters
          </div>

          <div style={{ display:'grid', gap:'1.25rem' }}>
            <div className="form-group">
              <label className="form-label">Agent</label>
              <select className="form-select" value={form.agent} onChange={e => set('agent', e.target.value)}>
                <option value="DQN" disabled={!status?.dqn_trained}>DQN {!status?.dqn_trained ? '(not trained)' : ''}</option>
                <option value="PPO" disabled={!status?.ppo_trained}>PPO {!status?.ppo_trained ? '(not trained)' : ''}</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Passengers — {form.pax} pax</label>
              <div className="slider-row">
                <input type="range" className="form-slider" min={50} max={220} step={1}
                  value={form.pax} onChange={e => set('pax', Number(e.target.value))} />
                <span className="slider-value">{form.pax}</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Weather Severity — {weatherLabel}</label>
              <div className="slider-row">
                <input type="range" className="form-slider" min={0} max={1} step={0.01}
                  value={form.weather} onChange={e => set('weather', parseFloat(e.target.value))} />
                <span className="slider-value">{form.weather.toFixed(2)}</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Fuel Price — ${form.fuel_price.toFixed(2)} / kg</label>
              <div className="slider-row">
                <input type="range" className="form-slider" min={0.3} max={2.0} step={0.01}
                  value={form.fuel_price} onChange={e => set('fuel_price', parseFloat(e.target.value))} />
                <span className="slider-value">${form.fuel_price.toFixed(2)}</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Distance — {form.distance.toLocaleString()} km</label>
              <div className="slider-row">
                <input type="range" className="form-slider" min={200} max={12000} step={50}
                  value={form.distance} onChange={e => set('distance', Number(e.target.value))} />
                <span className="slider-value">{form.distance.toLocaleString()}</span>
              </div>
            </div>

            <button
              id="btn-run-inference"
              className="btn btn-primary"
              style={{ width:'100%', justifyContent:'center', padding:'14px' }}
              onClick={handleInfer}
              disabled={loading || !canInfer}
            >
              {loading ? <><span className="spinner"/> Running Inference…</> : '🚀 Run Inference'}
            </button>

            {!canInfer && (
              <p style={{ color:'var(--warning)', fontSize:'0.78rem', textAlign:'center' }}>
                ⚠️ Train at least one RL agent first.
              </p>
            )}
            {error && (
              <p style={{ color:'var(--danger)', fontSize:'0.78rem' }}>✗ {error}</p>
            )}
          </div>
        </div>

        {/* Results */}
        <div style={{ display:'grid', gap:'1rem', alignContent:'start' }}>
          {result ? (
            <>
              <div className="card" style={{ background:'linear-gradient(135deg, rgba(16,185,129,0.08), rgba(59,130,246,0.05))' }}>
                <div className="card-title" style={{ color:'var(--success)' }}>
                  ✓ AI Recommendation — {form.agent}
                </div>

                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1rem', marginBottom:'1rem' }}>
                  <ResultCard
                    label="Fuel Uplift"
                    value={`${result.fuel_uplift_kg?.toLocaleString()} kg`}
                    sub={`${((result.fuel_uplift_kg / 25000) * 100).toFixed(1)}% of tank capacity`}
                    color="var(--primary)"
                  />
                  <ResultCard
                    label="Cargo Allocation"
                    value={`${result.cargo_kg?.toLocaleString()} kg`}
                    sub={`${((result.cargo_kg / 20000) * 100).toFixed(1)}% of max cargo`}
                    color="var(--accent)"
                  />
                </div>

                <div className="divider" />

                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'0.75rem' }}>
                  <div>
                    <div className="stat-label">Predicted Burn</div>
                    <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.95rem', color:'var(--text-100)' }}>
                      {result.actual_burn?.toLocaleString()} kg
                    </div>
                  </div>
                  <div>
                    <div className="stat-label">Revenue</div>
                    <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.95rem', color:'var(--success)' }}>
                      ${result.revenue?.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="stat-label">Fuel Cost</div>
                    <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.95rem', color:'var(--warning)' }}>
                      ${result.fuel_cost?.toLocaleString()}
                    </div>
                  </div>
                </div>

                <div className="divider" />

                <div style={{ textAlign:'center' }}>
                  <div className="stat-label" style={{ marginBottom:'6px' }}>Estimated Net Profit</div>
                  <div style={{
                    fontSize:'2rem',
                    fontWeight:800,
                    color: result.profit > 0 ? 'var(--success)' : 'var(--danger)',
                  }}>
                    ${result.profit?.toLocaleString()}
                  </div>
                  {result.penalty > 0 && (
                    <div style={{ color:'var(--danger)', fontSize:'0.78rem', marginTop:'4px' }}>
                      Penalty: ${result.penalty?.toLocaleString()}
                    </div>
                  )}
                </div>
              </div>

              {/* Visual: fuel vs burn */}
              <div className="card">
                <div className="card-title">⛽ Fuel Buffer Analysis</div>
                <div style={{ marginBottom:'8px', display:'flex', justifyContent:'space-between', fontSize:'0.8rem', color:'var(--text-300)' }}>
                  <span>Fuel Uplift</span>
                  <span>{result.fuel_uplift_kg?.toLocaleString()} kg</span>
                </div>
                <div className="progress-bar-wrap" style={{ height:'10px', marginBottom:'12px' }}>
                  <div className="progress-bar-fill" style={{ width:`${Math.min(100,(result.fuel_uplift_kg/25000)*100)}%` }} />
                </div>
                <div style={{ marginBottom:'8px', display:'flex', justifyContent:'space-between', fontSize:'0.8rem', color:'var(--text-300)' }}>
                  <span>Actual Burn</span>
                  <span>{result.actual_burn?.toLocaleString()} kg</span>
                </div>
                <div className="progress-bar-wrap" style={{ height:'10px' }}>
                  <div className="progress-bar-fill" style={{
                    width:`${Math.min(100,(result.actual_burn/25000)*100)}%`,
                    background:'linear-gradient(90deg, var(--warning), var(--danger))',
                  }} />
                </div>
                <div style={{ marginTop:'12px', fontSize:'0.78rem', color:'var(--text-400)' }}>
                  Buffer: {Math.max(0, result.fuel_uplift_kg - result.actual_burn).toLocaleString()} kg
                  {result.fuel_uplift_kg < result.actual_burn
                    ? <span style={{ color:'var(--danger)', marginLeft:'8px' }}>⚠️ FUEL SHORTAGE!</span>
                    : <span style={{ color:'var(--success)', marginLeft:'8px' }}>✓ Safe margin</span>
                  }
                </div>
              </div>
            </>
          ) : (
            <div className="card" style={{ textAlign:'center', padding:'4rem', color:'var(--text-400)' }}>
              <div style={{ fontSize:'3rem', marginBottom:'1rem' }}>🛫</div>
              <div style={{ fontWeight:600, marginBottom:'4px' }}>Ready for inference</div>
              <div style={{ fontSize:'0.82rem' }}>Configure flight parameters and click Run Inference.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
