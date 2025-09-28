import React, { useEffect, useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type OutputItem = {
  path: string
  size: number
  mtime: number
}

export function App() {
  const [items, setItems] = useState<OutputItem[]>([])
  const [loading, setLoading] = useState(false)
  const [modelId, setModelId] = useState('Qwen/Qwen3-1.7B-Base')
  const [adapterId, setAdapterId] = useState('')
  const [inputPath, setInputPath] = useState('data/elyza-tasks-100-TV_0.jsonl')
  const [outputPath, setOutputPath] = useState('outputs/ui_run.jsonl')
  const [useUnsloth, setUseUnsloth] = useState(false)
  const [maxNewTokens, setMaxNewTokens] = useState(512)
  const [message, setMessage] = useState('')

  const listOutputs = async () => {
    const res = await fetch(`${API_BASE}/api/outputs`)
    const data = await res.json()
    setItems(data.items || [])
  }

  useEffect(() => {
    listOutputs()
  }, [])

  const onRun = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setMessage('')
    try {
      const res = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, adapter_id: adapterId || undefined, input_path: inputPath, output_path: outputPath, use_unsloth: useUnsloth, max_new_tokens: maxNewTokens }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || 'failed')
      setMessage(`started: ${data.output}`)
      await listOutputs()
    } catch (err: any) {
      setMessage(`error: ${String(err.message || err)}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h1>LLM Experiments Dashboard (TS)</h1>

      <form onSubmit={onRun} style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8, marginBottom: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <label>
            <div>Model ID</div>
            <input value={modelId} onChange={e => setModelId(e.target.value)} style={{ width: '100%' }} />
          </label>
          <label>
            <div>Adapter ID (LoRA)</div>
            <input value={adapterId} onChange={e => setAdapterId(e.target.value)} style={{ width: '100%' }} />
          </label>
          <label>
            <div>Input JSONL</div>
            <input value={inputPath} onChange={e => setInputPath(e.target.value)} style={{ width: '100%' }} />
          </label>
          <label>
            <div>Output JSONL</div>
            <input value={outputPath} onChange={e => setOutputPath(e.target.value)} style={{ width: '100%' }} />
          </label>
          <label>
            <div>Use Unsloth</div>
            <select value={useUnsloth ? 'true' : 'false'} onChange={e => setUseUnsloth(e.target.value === 'true')}>
              <option value='false'>false</option>
              <option value='true'>true</option>
            </select>
          </label>
          <label>
            <div>Max New Tokens</div>
            <input type='number' value={maxNewTokens} onChange={e => setMaxNewTokens(parseInt(e.target.value || '0', 10) || 0)} />
          </label>
        </div>
        <div style={{ marginTop: 12 }}>
          <button type='submit' disabled={loading}>{loading ? 'Running...' : 'Run'}</button>
          <span style={{ marginLeft: 8, color: '#666' }}>Backend: {API_BASE}</span>
        </div>
        {message && <div style={{ marginTop: 8 }}>{message}</div>}
      </form>

      <div style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8 }}>
        <h3>Outputs</h3>
        <table style={{ borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr>
              <th style={{ border: '1px solid #ccc', padding: 6, textAlign: 'left' }}>#</th>
              <th style={{ border: '1px solid #ccc', padding: 6, textAlign: 'left' }}>path</th>
              <th style={{ border: '1px solid #ccc', padding: 6, textAlign: 'left' }}>action</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => (
              <tr key={it.path}>
                <td style={{ border: '1px solid #ccc', padding: 6 }}>{i + 1}</td>
                <td style={{ border: '1px solid #ccc', padding: 6 }}>{it.path}</td>
                <td style={{ border: '1px solid #ccc', padding: 6 }}>
                  <a href={`${API_BASE}/api/view?f=${encodeURIComponent(it.path)}`} target="_blank" rel="noreferrer">view</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
