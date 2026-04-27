/**
 * API client — communicates with the FastAPI backend
 */
const BASE = 'http://localhost:8000'

async function _json(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`)
  return res.json()
}

export const api = {
  status:             ()                              => _json('/api/status'),
  dataSample:         (n = 20)                        => _json(`/api/data/sample?n=${n}`),
  metrics:            ()                              => _json('/api/metrics'),
  trainRegression:    (n_samples = 10000)             =>
    _json('/api/train/regression', { method: 'POST',
      body: JSON.stringify({ n_samples }) }),
  trainRL:            (agent, n_episodes)             =>
    _json('/api/train/rl', { method: 'POST',
      body: JSON.stringify({ agent, n_episodes }) }),
  infer:              (payload)                       =>
    _json('/api/infer', { method: 'POST', body: JSON.stringify(payload) }),
}

export function createWS(onMessage) {
  const ws = new WebSocket('ws://localhost:8000/ws/training')
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch (_) {}
  }
  return ws
}
