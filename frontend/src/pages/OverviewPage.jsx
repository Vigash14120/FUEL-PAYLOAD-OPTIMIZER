/**
 * OverviewPage — pipeline diagram + key stats + quick actions
 */
import { useEffect, useState } from 'react'
import { api } from '../api'

function PipelineNode({ title, sub, state }) {
  const cls = state === 'done' ? 'done' : state === 'active' ? 'active' : ''
  return (
    <div className={`pipeline-node ${cls}`}>
      <div className="pipeline-node-title">{title}</div>
      <div className="pipeline-node-sub">{sub}</div>
    </div>
  )
}

export default function OverviewPage({ status, metrics, onTabChange }) {
  const regDone = status?.regression_trained
  const dqnDone = status?.dqn_trained
  const ppoDone = status?.ppo_trained

  return (
    <div>
      {/* Hero */}
      <div className="hero">
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', flexWrap:'wrap', gap:'1rem' }}>
          <div>
            <h1 className="section-heading" style={{ fontSize:'1.6rem', marginBottom:'0.5rem' }}>
              ✈️ Fuel Payload Optimizer
            </h1>
            <p style={{ color:'var(--text-300)', maxWidth:'540px', lineHeight:1.6 }}>
              Deep Reinforcement Learning system combining SVM / Random Forest regression
              models with DQN and PPO agents to optimize airline fuel uplift and cargo
              allocation decisions.
            </p>
          </div>
          <div style={{ display:'flex', gap:'8px', flexWrap:'wrap' }}>
            <span className="badge badge-blue">DQN</span>
            <span className="badge badge-purple">PPO</span>
            <span className="badge badge-green">SVM / RF</span>
          </div>
        </div>

        {/* Pipeline Diagram */}
        <div style={{ marginTop:'2rem' }}>
          <div style={{ fontSize:'0.75rem', color:'var(--text-400)', marginBottom:'12px', textTransform:'uppercase', letterSpacing:'0.06em' }}>
            System Pipeline
          </div>
          <div className="pipeline">
            <PipelineNode title="Flight Data Logs" sub="Synthetic + Historical" state="done" />
            <span className="pipeline-arrow">→</span>
            <PipelineNode title="Regression Model" sub="SVM / Random Forest" state={regDone ? 'done' : ''} />
            <span className="pipeline-arrow">→</span>
            <PipelineNode title="Simulation Env" sub="Gymnasium-compatible" state={regDone ? 'done' : ''} />
            <span className="pipeline-arrow">⇄</span>
            <PipelineNode title="Deep RL Agent" sub="DQN / PPO" state={(dqnDone || ppoDone) ? 'done' : regDone ? 'active' : ''} />
          </div>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid-4" style={{ marginBottom:'1.5rem' }}>
        <div className="stat-card blue">
          <span className="stat-icon">🔬</span>
          <div className="stat-label">Regression Model</div>
          <div className="stat-value" style={{ color: regDone ? 'var(--success)' : 'var(--text-400)', fontSize:'1.1rem' }}>
            {regDone ? 'Trained ✓' : 'Untrained'}
          </div>
          <div className="stat-sub">
            {metrics?.regression ? `Best: ${metrics.regression.best}` : 'SVM / Random Forest'}
          </div>
        </div>

        <div className="stat-card cyan">
          <span className="stat-icon">🧠</span>
          <div className="stat-label">DQN Agent</div>
          <div className="stat-value" style={{ color: dqnDone ? 'var(--success)' : 'var(--text-400)', fontSize:'1.1rem' }}>
            {dqnDone ? 'Trained ✓' : 'Untrained'}
          </div>
          <div className="stat-sub">
            {metrics?.dqn ? `Avg Profit: $${metrics.dqn.final_avg_profit?.toLocaleString()}` : 'Deep Q-Network'}
          </div>
        </div>

        <div className="stat-card purple">
          <span className="stat-icon">🎯</span>
          <div className="stat-label">PPO Agent</div>
          <div className="stat-value" style={{ color: ppoDone ? 'var(--success)' : 'var(--text-400)', fontSize:'1.1rem' }}>
            {ppoDone ? 'Trained ✓' : 'Untrained'}
          </div>
          <div className="stat-sub">
            {metrics?.ppo ? `Avg Profit: $${metrics.ppo.final_avg_profit?.toLocaleString()}` : 'Proximal Policy Opt.'}
          </div>
        </div>

        <div className="stat-card green">
          <span className="stat-icon">📈</span>
          <div className="stat-label">Best R² Score</div>
          <div className="stat-value">
            {metrics?.regression
              ? Object.values(metrics.regression.metrics || {})
                  .map(m => m.r2)
                  .reduce((a, b) => Math.max(a, b), 0)
                  .toFixed(4)
              : '—'}
          </div>
          <div className="stat-sub">Regression accuracy</div>
        </div>
      </div>

      {/* Quick Start Guide */}
      <div className="card">
        <div className="card-title">🚀 Quick Start Guide</div>
        <div style={{ display:'grid', gap:'12px' }}>
          {[
            { step:'1', label:'Train Regression Model', desc:'Go to Training → Regression tab. Generate flight data & train SVM + Random Forest.', tab:'training', done: regDone },
            { step:'2', label:'Train RL Agent (DQN or PPO)', desc:'After regression is trained, train the RL agent for optimal fuel & cargo decisions.', tab:'training', done: dqnDone || ppoDone },
            { step:'3', label:'Run Flight Inference', desc:'Go to Inference tab. Set flight parameters and get AI-recommended fuel uplift & cargo.', tab:'inference', done: false },
            { step:'4', label:'Analyze Training Curves', desc:'Review reward curves, profit trends, and model metrics on the Metrics tab.', tab:'metrics', done: false },
          ].map(({ step, label, desc, tab, done }) => (
            <div key={step} style={{
              display:'flex', gap:'12px', alignItems:'flex-start',
              padding:'12px', borderRadius:'var(--radius-sm)',
              background: done ? 'rgba(16,185,129,0.05)' : 'var(--bg-800)',
              border: `1px solid ${done ? 'rgba(16,185,129,0.2)' : 'var(--border)'}`,
            }}>
              <div style={{
                width:'28px', height:'28px', borderRadius:'50%',
                background: done ? 'var(--success)' : 'var(--bg-600)',
                display:'flex', alignItems:'center', justifyContent:'center',
                fontSize:'0.75rem', fontWeight:'700', flexShrink:0,
                color: done ? 'white' : 'var(--text-400)',
              }}>
                {done ? '✓' : step}
              </div>
              <div>
                <div style={{ fontWeight:600, color:'var(--text-100)', fontSize:'0.85rem', marginBottom:'2px' }}>{label}</div>
                <div style={{ color:'var(--text-400)', fontSize:'0.78rem' }}>{desc}</div>
              </div>
              <button
                className="btn btn-secondary"
                style={{ marginLeft:'auto', padding:'6px 14px', fontSize:'0.78rem' }}
                onClick={() => onTabChange(tab)}
              >
                Go →
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
