/**
 * DataPage — sample flight data table
 */
import { useState, useEffect } from 'react'
import { api } from '../api'

const COLS = [
  { key:'pax',        label:'PAX',          fmt: v => v },
  { key:'weather',    label:'Weather',       fmt: v => v.toFixed(2) },
  { key:'fuel_price', label:'Fuel ($/kg)',   fmt: v => `$${v.toFixed(2)}` },
  { key:'distance',   label:'Distance (km)', fmt: v => v.toLocaleString() },
  { key:'cargo_kg',   label:'Cargo (kg)',    fmt: v => v.toLocaleString() },
  { key:'fuel_burn',  label:'Burn (kg)',     fmt: v => v.toLocaleString() },
  { key:'revenue',    label:'Revenue',       fmt: v => `$${v.toLocaleString()}` },
  { key:'fuel_cost',  label:'Fuel Cost',     fmt: v => `$${v.toLocaleString()}` },
  { key:'profit',     label:'Profit',        fmt: v => `$${v.toLocaleString()}`, color: v => v >= 0 ? 'var(--success)' : 'var(--danger)' },
]

export default function DataPage() {
  const [rows,    setRows]    = useState([])
  const [loading, setLoading] = useState(false)
  const [n,       setN]       = useState(30)

  const fetchData = async () => {
    setLoading(true)
    try { setRows(await api.dataSample(n)) } catch (_) {}
    finally { setLoading(false) }
  }

  useEffect(() => { fetchData() }, [])

  return (
    <div style={{ display:'grid', gap:'1.5rem' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:'1rem' }}>
        <div>
          <h2 className="section-heading">Flight Data Explorer</h2>
          <p className="section-sub">Synthetic flight log records used for regression model training.</p>
        </div>
        <div style={{ display:'flex', gap:'8px', alignItems:'center' }}>
          <select className="form-select" style={{ width:'auto' }} value={n} onChange={e => setN(Number(e.target.value))}>
            {[20,50,100,200].map(v => <option key={v} value={v}>{v} rows</option>)}
          </select>
          <button className="btn btn-primary" onClick={fetchData} disabled={loading}>
            {loading ? <span className="spinner"/> : '↻'} Refresh
          </button>
        </div>
      </div>

      <div className="card" style={{ padding:0, overflow:'hidden' }}>
        <div style={{ overflowX:'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                {COLS.map(c => <th key={c.key}>{c.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  <td style={{ color:'var(--text-400)' }}>{i + 1}</td>
                  {COLS.map(c => (
                    <td key={c.key} style={{ color: c.color ? c.color(row[c.key]) : 'var(--text-200)' }}>
                      {c.fmt(row[c.key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Schema */}
      <div className="card">
        <div className="card-title">📐 Feature Schema</div>
        <div className="grid-3" style={{ gap:'1rem' }}>
          {[
            { name:'pax',        type:'int',   range:'50–220',    desc:'Passenger count' },
            { name:'weather',    type:'float', range:'0.0–1.0',   desc:'0=clear, 1=severe storm' },
            { name:'fuel_price', type:'float', range:'$0.50–1.50',desc:'USD per kg of Jet-A fuel' },
            { name:'distance',   type:'float', range:'200–8,000 km',desc:'Great-circle route distance' },
            { name:'cargo_kg',   type:'float', range:'500–15,000 kg',desc:'Payload cargo weight' },
            { name:'fuel_burn',  type:'float', range:'~600–25,000 kg',desc:'Target variable (physics model)' },
          ].map(f => (
            <div key={f.name} style={{
              padding:'12px', borderRadius:'var(--radius-sm)',
              background:'var(--bg-800)', border:'1px solid var(--border)',
            }}>
              <div style={{ display:'flex', gap:'8px', marginBottom:'4px', alignItems:'center' }}>
                <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.8rem', color:'var(--primary)' }}>{f.name}</span>
                <span className="badge badge-amber" style={{ fontSize:'0.65rem' }}>{f.type}</span>
              </div>
              <div style={{ fontSize:'0.72rem', color:'var(--text-400)', marginBottom:'4px' }}>{f.range}</div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-300)' }}>{f.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
