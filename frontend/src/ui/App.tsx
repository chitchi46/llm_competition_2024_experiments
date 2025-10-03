import React, { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type OutputItem = {
  path: string
  size: number
  mtime: number
}

type EvalRun = {
  run_id: string
  status: string
  total: number
  done: number
  avg_score: number
  created_at: number
  elapsed: number
}

type EvalStatus = {
  run_id: string
  status: string
  total: number
  done: number
  failed: number
  cached_hits: number
  avg_score: number
  elapsed: number
  eta: number
  error: string | null
}

type DryRunEstimate = {
  total: number
  avg_tokens: number
  qps: number
  estimated_time_sec: number
  estimated_time_min: number
  estimated_cost_usd: number
  error?: string
}

export function App() {
  const [items, setItems] = useState<OutputItem[]>([])
  const [runs, setRuns] = useState<EvalRun[]>([])
  const [activeRun, setActiveRun] = useState<EvalStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  
  // Gemini設定
  const [geminiMasked, setGeminiMasked] = useState('')
  const [geminiModel, setGeminiModel] = useState('')
  const [geminiInput, setGeminiInput] = useState('')
  const [geminiConfigured, setGeminiConfigured] = useState(false)
  const [geminiValid, setGeminiValid] = useState<boolean | null>(null)
  const [geminiReason, setGeminiReason] = useState('')

  // 評価ラン設定
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [qps, setQps] = useState('1.0')
  const [promptVersion, setPromptVersion] = useState('v1')
  const [cache, setCache] = useState('on')
  const [dryRunResult, setDryRunResult] = useState<DryRunEstimate | null>(null)

  const listOutputs = async () => {
    const res = await fetch(`${API_BASE}/api/outputs`)
    const data = await res.json()
    setItems(data.items || [])
  }

  const listRuns = async () => {
    const res = await fetch(`${API_BASE}/api/eval_runs`)
    const data = await res.json()
    setRuns(data.runs || [])
  }

  const checkGeminiStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/gemini_status`)
      const data = await res.json()
      if (typeof data?.configured === 'boolean') setGeminiConfigured(!!data.configured)
      if (typeof data?.valid === 'boolean') setGeminiValid(!!data.valid)
      if (typeof data?.reason === 'string') setGeminiReason(data.reason)
      if (data?.masked) setGeminiMasked(data.masked)
      if (data?.model) setGeminiModel(data.model)
      if (data?.configured === true && data?.valid === false) {
        setMessage(`Gemini API Key が無効の可能性: ${data?.reason || 'invalid'}`)
      }
    } catch {}
  }

  useEffect(() => {
    listOutputs()
    listRuns()
    checkGeminiStatus()
  }, [])

  const saveGeminiKey = async () => {
    if (!geminiInput) return
    setLoading(true)
    setMessage('')
    try {
      const res = await fetch(`${API_BASE}/api/gemini_key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: geminiInput })
      })
      let data: any = {}
      try { data = await res.json() } catch {}
      if (!res.ok) {
        const hid = res.headers.get('X-Error-Id') || ''
        if (hid) console.warn('error-id:', hid)
        const reason = data?.reason ? `: ${data.reason}` : ''
        throw new Error(`${data?.error || 'failed'}${reason}`)
      }
      if (data.masked) setGeminiMasked(data.masked)
      setGeminiConfigured(true)
      if (typeof data?.valid === 'boolean') setGeminiValid(!!data.valid)
      if (typeof data?.reason === 'string') setGeminiReason(data.reason)
      setGeminiInput('')
      await checkGeminiStatus()
      setMessage(data.valid === false ? `保存しましたが無効: ${data.reason || 'invalid'}` : 'Gemini API Key を設定しました。')
    } catch (e: any) {
      setMessage(`保存失敗: ${String(e.message || e)}`)
    } finally {
      setLoading(false)
    }
  }

  const onDryRun = async () => {
    if (!selectedFile) {
      setMessage('ファイルを選択してください')
      return
    }
    setLoading(true)
    setMessage('')
    setDryRunResult(null)
    try {
      const fd = new FormData()
      fd.append('file', selectedFile)
      fd.append('qps', qps)
      fd.append('prompt_version', promptVersion)
      fd.append('cache', cache)
      fd.append('dry_run', 'true')
      
      const res = await fetch(`${API_BASE}/api/eval_start`, {
        method: 'POST',
        body: fd,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || 'dry_run failed')
      
      if (data.estimate?.error) {
        setMessage(`ドライラン失敗: ${data.estimate.error}`)
      } else {
        setDryRunResult(data.estimate)
        setMessage('ドライラン完了。問題なければ「実行」ボタンで本番開始してください。')
      }
    } catch (err: any) {
      setMessage(`error: ${String(err.message || err)}`)
    } finally {
      setLoading(false)
    }
  }

  const onRealRun = async () => {
    if (!selectedFile) {
      setMessage('ファイルを選択してください')
      return
    }
    if (!dryRunResult) {
      setMessage('先にドライランを実行してください')
      return
    }
    setLoading(true)
    setMessage('')
    try {
      const fd = new FormData()
      fd.append('file', selectedFile)
      fd.append('qps', qps)
      fd.append('prompt_version', promptVersion)
      fd.append('cache', cache)
      fd.append('dry_run', 'false')
      
      const res = await fetch(`${API_BASE}/api/eval_start`, {
        method: 'POST',
        body: fd,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || 'start failed')
      
      setMessage(`評価開始: run_id=${data.run_id}`)
      setActiveRun({ run_id: data.run_id, status: 'running', total: dryRunResult.total, done: 0, failed: 0, cached_hits: 0, avg_score: 0, elapsed: 0, eta: 0, error: null })
      setDryRunResult(null)
      setSelectedFile(null)
      await listRuns()
    } catch (err: any) {
      setMessage(`error: ${String(err.message || err)}`)
    } finally {
      setLoading(false)
    }
  }

  const onCancelRun = async (run_id: string) => {
    try {
      const fd = new FormData()
      fd.append('run_id', run_id)
      const res = await fetch(`${API_BASE}/api/eval_cancel`, {
        method: 'POST',
        body: fd,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || 'cancel failed')
      setMessage(`キャンセルしました: ${run_id}`)
      if (activeRun?.run_id === run_id) {
        setActiveRun(null)
      }
      await listRuns()
    } catch (err: any) {
      setMessage(`error: ${String(err.message || err)}`)
    }
  }

  // 進捗ポーリング
  useEffect(() => {
    if (!activeRun || activeRun.status !== 'running') return
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/eval_status?run_id=${activeRun.run_id}`)
        const data = await res.json()
        if (res.ok) {
          setActiveRun(data)
          if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
            setMessage(`評価${data.status}: ${activeRun.run_id} (平均スコア: ${data.avg_score})`)
            await listRuns()
            await listOutputs()
            setTimeout(() => setActiveRun(null), 3000)
          }
        }
      } catch {}
    }, 2000)
    
    return () => clearInterval(interval)
  }, [activeRun])

  const formatTime = (sec: number) => {
    if (sec < 60) return `${Math.round(sec)}秒`
    const min = Math.floor(sec / 60)
    const s = Math.round(sec % 60)
    return `${min}分${s}秒`
  }

  const formatDate = (ts: number) => {
    return new Date(ts * 1000).toLocaleString('ja-JP')
  }

  return (
    <div className="container">
      <h1 className="page-title">LLM コンペ評価ダッシュボード</h1>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginTop: 0 }}>このページはコンペの評価用です</h3>
        <ul style={{ margin: '8px 0 0 18px' }}>
          <li>Gemini API Key を入力し「保存」します（表示: <b>{geminiConfigured ? '入力済' : '未入力'}</b>）。</li>
          <li><code>.jsonl</code> / <code>.txt</code> をアップロードし、「ドライラン」で見積→「実行」で本番採点します。</li>
          <li>全件採点が既定です。進捗を監視し、必要に応じてキャンセルできます。</li>
        </ul>
      </div>

      {/* Gemini API Key */}
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <div>
          <div className="muted">Gemini API</div>
          <div>API Key: {geminiConfigured ? '入力済' : '未入力'} {geminiModel && <span className="muted">({geminiModel})</span>}</div>
          {geminiConfigured && geminiValid === false && (
            <div className="muted" style={{ color: '#e57373' }}>無効の可能性: {geminiReason || 'invalid'}</div>
          )}
          {geminiConfigured && geminiValid === true && (
            <div className="muted" style={{ color: '#66bb6a' }}>有効</div>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <input type="password" placeholder="Paste API Key" value={geminiInput} onChange={e => setGeminiInput(e.target.value)} />
          <button className="btn" onClick={saveGeminiKey} disabled={loading || !geminiInput}>{loading ? '保存中...' : '保存'}</button>
        </div>
      </div>

      {/* ラン開始パネル */}
      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>評価ラン開始</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <label>ファイル (.jsonl / .txt)</label><br/>
            <input type="file" accept=".jsonl,.txt,.json" onChange={e => setSelectedFile(e.target.files?.[0] || null)} disabled={loading} />
          </div>
          <div>
            <label>QPS</label><br/>
            <input type="number" step="0.1" value={qps} onChange={e => setQps(e.target.value)} disabled={loading} />
          </div>
          <div>
            <label>プロンプト版</label><br/>
            <input type="text" value={promptVersion} onChange={e => setPromptVersion(e.target.value)} disabled={loading} />
          </div>
          <div>
            <label>キャッシュ</label><br/>
            <select value={cache} onChange={e => setCache(e.target.value)} disabled={loading}>
              <option value="on">有効</option>
              <option value="off">無効</option>
            </select>
          </div>
        </div>
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          <button className="btn" onClick={onDryRun} disabled={loading || !selectedFile}>ドライラン（見積）</button>
          <button className="btn" onClick={onRealRun} disabled={loading || !dryRunResult} style={{ backgroundColor: dryRunResult ? '#4caf50' : undefined }}>実行（本番）</button>
        </div>
        {dryRunResult && !dryRunResult.error && (
          <div style={{ marginTop: 12, padding: 8, backgroundColor: '#f0f8ff', borderRadius: 4 }}>
            <strong>見積結果:</strong> 全{dryRunResult.total}件、
            平均{dryRunResult.avg_tokens}トークン/件、
            所要時間 約{formatTime(dryRunResult.estimated_time_sec)}、
            概算費用 約${dryRunResult.estimated_cost_usd}
          </div>
        )}
      </div>

      {/* 進捗ビュー */}
      {activeRun && activeRun.status === 'running' && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3 style={{ marginTop: 0 }}>実行中: {activeRun.run_id}</h3>
          <div style={{ marginBottom: 8 }}>
            進捗: {activeRun.done} / {activeRun.total} 件
            {activeRun.cached_hits > 0 && ` (キャッシュ命中: ${activeRun.cached_hits})`}
          </div>
          <div style={{ width: '100%', height: 24, backgroundColor: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ width: `${(activeRun.done / activeRun.total) * 100}%`, height: '100%', backgroundColor: '#4caf50', transition: 'width 0.3s' }}></div>
          </div>
          <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between' }}>
            <div>経過: {formatTime(activeRun.elapsed)}</div>
            <div>残り: {formatTime(activeRun.eta)}</div>
            <div>平均スコア: {activeRun.avg_score.toFixed(2)}</div>
          </div>
          <button className="btn" onClick={() => onCancelRun(activeRun.run_id)} style={{ marginTop: 12, backgroundColor: '#f44336' }}>キャンセル</button>
        </div>
      )}

      {message && <div className="card" style={{ marginTop: 12 }}>{message}</div>}

      {/* ラン履歴 */}
      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>評価ラン履歴</h3>
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>状態</th>
              <th>件数</th>
              <th>平均スコア</th>
              <th>所要時間</th>
              <th>作成日時</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {runs.map(run => (
              <tr key={run.run_id}>
                <td>{run.run_id}</td>
                <td>{run.status}</td>
                <td>{run.done} / {run.total}</td>
                <td>{run.avg_score.toFixed(2)}</td>
                <td>{formatTime(run.elapsed)}</td>
                <td>{formatDate(run.created_at)}</td>
                <td>
                  <a href={`${API_BASE}/api/eval_result?run_id=${run.run_id}&format=jsonl`} target="_blank" rel="noreferrer">JSONL</a>
                  {' | '}
                  <a href={`${API_BASE}/api/eval_result?run_id=${run.run_id}&format=csv`} target="_blank" rel="noreferrer">CSV</a>
                  {run.status === 'running' && (
                    <> | <button onClick={() => onCancelRun(run.run_id)} style={{ padding: '2px 6px', fontSize: '0.9em' }}>キャンセル</button></>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 旧Outputs（参考） */}
      <div className="panel card" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>Outputs（参考）</h3>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>path</th>
              <th>action</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => (
              <tr key={it.path}>
                <td>{i + 1}</td>
                <td>{it.path}</td>
                <td>
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
