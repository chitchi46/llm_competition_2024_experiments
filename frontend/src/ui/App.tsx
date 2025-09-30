import React, { useEffect, useState } from 'react'

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
  const [geminiMasked, setGeminiMasked] = useState('')
  const [geminiModel, setGeminiModel] = useState('')
  const [geminiInput, setGeminiInput] = useState('')
  const [geminiConfigured, setGeminiConfigured] = useState(false)
  const [geminiValid, setGeminiValid] = useState<boolean | null>(null)
  const [geminiReason, setGeminiReason] = useState('')

  const listOutputs = async () => {
    const res = await fetch(`${API_BASE}/api/outputs`)
    const data = await res.json()
    setItems(data.items || [])
  }

  useEffect(() => {
    listOutputs()
    ;(async () => {
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
    })()
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
        // IDはログにのみ残し、UIには表示しない。ヘッダにあればコンソールへ。
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
      // 直後に有効性チェック
      try {
        const chk = await fetch(`${API_BASE}/api/gemini_status`)
        const cd = await chk.json()
        if (typeof cd?.valid === 'boolean') setGeminiValid(!!cd.valid)
        if (typeof cd?.reason === 'string') setGeminiReason(cd.reason)
        if (cd?.valid === false) {
          setMessage(`保存しましたが、Gemini API Key が無効です: ${cd?.reason || 'invalid'}`)
        } else {
          setMessage('Gemini API Key を設定しました。')
        }
      } catch {
        setMessage(data.valid === false ? `保存しましたが無効: ${data.reason || 'invalid'}` : 'Gemini API Key を設定しました。')
      }
    } catch (e: any) {
      setMessage(`保存失敗: ${String(e.message || e)}`)
    } finally {
      setLoading(false)
    }
  }

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

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setMessage('')
    setLoading(true)
    try {
      const fd = new FormData()
      fd.append('file', f)
      const res = await fetch(`${API_BASE}/api/eval_upload`, {
        method: 'POST',
        body: fd,
      })
      let data: any = {}
      try { data = await res.json() } catch {}
      if (!res.ok) {
        const hid = res.headers.get('X-Error-Id') || ''
        if (hid) console.warn('error-id:', hid)
        throw new Error(`${data?.error || 'upload failed'}`)
      }
      setMessage(`Gemini評価 完了: avg=${(data.avg_score ?? 0).toFixed(2)} count=${data.count}. 出力: ${data.output}`)
      await listOutputs()
    } catch (err: any) {
      setMessage(`error: ${String(err.message || err)}`)
    } finally {
      setLoading(false)
      e.currentTarget.value = ''
    }
  }

  return (
    <div className="container">
      <h1 className="page-title">LLM コンペ評価ダッシュボード</h1>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginTop: 0 }}>このページはコンペの評価用です</h3>
        <ul style={{ margin: '8px 0 0 18px' }}>
          <li>上で Gemini API Key を入力し「保存」できます（表示は <b>{geminiConfigured ? '入力済' : '未入力'}</b>）。</li>
          <li><code>.jsonl</code> / <code>.txt</code> をアップロードすると、Gemini が自動採点し平均スコアと出力パスを表示します。</li>
          <li>下の Outputs から生成された <code>outputs/*.jsonl</code> を開いて内容を確認できます。</li>
        </ul>
      </div>

      {/* 推論フォームは非表示（評価のみ運用） */}
      {/* <form onSubmit={onRun} className="card"> ... </form> */}

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


      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>評価用ファイルをアップロード（.jsonl / .txt）</h3>
        <p className="muted">.txt の場合は各行を 1 レコードとして評価します（最大50件）。</p>
        <input type="file" accept=".jsonl,.txt" onChange={onUpload} disabled={loading} />
        {loading && <div className="muted" style={{ marginTop: 8 }}>評価実行中...</div>}
      </div>

      {message && <div className="card" style={{ marginTop: 12 }}>{message}</div>}

      <div className="panel card">
        <h3 style={{ marginTop: 0 }}>Outputs</h3>
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
