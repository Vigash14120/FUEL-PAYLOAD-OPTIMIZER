import { useState, useEffect, useRef } from 'react'
import { api, createWS } from './api'
import OverviewPage   from './pages/OverviewPage'
import TrainingPage   from './pages/TrainingPage'
import MetricsPage    from './pages/MetricsPage'
import InferencePage  from './pages/InferencePage'
import DataPage       from './pages/DataPage'

const TABS = [
  { id:'overview',   label:'Overview',   icon:'🏠' },
  { id:'training',   label:'Training',   icon:'🔬' },
  { id:'metrics',    label:'Metrics',    icon:'📊' },
  { id:'inference',  label:'Inference',  icon:'🛫' },
  { id:'data',       label:'Data',       icon:'🗃️' },
]

export default function App() {
  const [tab,        setTab]        = useState('overview')
  const [status,     setStatus]     = useState(null)
  const [metrics,    setMetrics]    = useState(null)
  const [wsMsg,      setWsMsg]      = useState(null)
  const [connected,  setConnected]  = useState(false)
  const wsRef = useRef(null)

  const fetchStatus = async () => {
    try {
      const s = await api.status()
      setStatus(s)
    } catch (_) {
      setStatus(null)
    }
  }

  const fetchMetrics = async () => {
    try { setMetrics(await api.metrics()) } catch (_) {}
  }

  const onStatusUpdate = () => { fetchStatus(); fetchMetrics() }

  // Connect WebSocket
  useEffect(() => {
    const connect = () => {
      try {
        const ws = createWS((msg) => {
          setWsMsg(msg)
          if (['regression_done', 'rl_done'].includes(msg.type)) {
            onStatusUpdate()
          }
        })
        ws.onopen  = () => setConnected(true)
        ws.onclose = () => {
          setConnected(false)
          setTimeout(connect, 3000)   // reconnect
        }
        ws.onerror = () => ws.close()
        wsRef.current = ws
      } catch (_) {}
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  // Poll status
  useEffect(() => {
    fetchStatus()
    fetchMetrics()
    const id = setInterval(fetchStatus, 5000)
    return () => clearInterval(id)
  }, [])

  const isTraining = status?.training_running

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">✈️</div>
          <div>
            <div className="header-title">Fuel Payload Optimizer</div>
            <div className="header-subtitle">Deep Reinforcement Learning</div>
          </div>
        </div>
        <div className="header-status">
          <div className={`status-dot ${connected ? (isTraining ? 'training' : 'online') : ''}`} />
          <span>
            {connected
              ? isTraining ? 'Training…' : 'API Connected'
              : 'Connecting…'}
          </span>
        </div>
      </header>

      {/* Tabs */}
      <nav className="nav-tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`nav-tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
            id={`tab-${t.id}`}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </nav>

      {/* Main */}
      <main className="main">
        {tab === 'overview'  && <OverviewPage  status={status} metrics={metrics} onTabChange={setTab} />}
        {tab === 'training'  && <TrainingPage  status={status} onStatusUpdate={onStatusUpdate} wsMessages={wsMsg} />}
        {tab === 'metrics'   && <MetricsPage   />}
        {tab === 'inference' && <InferencePage status={status} />}
        {tab === 'data'      && <DataPage      />}
      </main>
    </div>
  )
}
