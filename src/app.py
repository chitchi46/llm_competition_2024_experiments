from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import json

from src.infer import run_inference


app = FastAPI(title="LLM Experiments Dashboard")

# CORS (for Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def render_index(rows):
    rows_html = "".join(
        f"<tr><td>{i+1}</td><td>{r['path']}</td>"
        f"<td><a href='/view?f={r['path']}' target='_blank'>view</a></td></tr>"
        for i, r in enumerate(rows)
    )
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>LLM Dashboard</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    input, select, button {{ padding: 6px 8px; }}
    .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 16px; }}
  </style>
  <meta name='viewport' content='width=device-width,initial-scale=1'/>
  <script>
    function onSubmitForm() {{
      document.getElementById('runBtn').disabled = true;
      return true;
    }}
  </script>
  <meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate' />
  <meta http-equiv='Pragma' content='no-cache' />
  <meta http-equiv='Expires' content='0' />
  <meta http-equiv='refresh' content='0;url=/ui' />
</head>
<body>
</body>
</html>
"""


def render_ui(model_id_default: str, adapter_id_default: str):
    # list outputs
    rows = []
    for p in sorted(OUTPUTS_DIR.glob("*.jsonl")):
        rows.append({"path": str(p.relative_to(BASE_DIR))})

    rows_html = "".join(
        f"<tr><td>{i+1}</td><td>{r['path']}</td>"
        f"<td><a href='/view?f={r['path']}' target='_blank'>view</a></td></tr>"
        for i, r in enumerate(rows)
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>LLM Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Hiragino Sans', 'Noto Sans JP', 'Helvetica Neue', Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    input, select, button {{ padding: 6px 8px; }}
    .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 16px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .row > div {{ flex: 1 1 320px; }}
    .muted {{ color: #666; font-size: 12px; }}
  </style>
  <meta name='viewport' content='width=device-width,initial-scale=1'/>
</head>
<body>
  <h1>LLM Experiments Dashboard</h1>

  <div class='card'>
    <h3>推論フォーム</h3>
    <form method='post' action='/run' onsubmit='return onSubmitForm()'>
      <div class='row'>
        <div>
          <label>Model ID</label><br/>
          <input type='text' name='model_id' value='{model_id_default}' style='width:100%'/>
          <div class='muted'>例: Qwen/Qwen3-1.7B-Base, google/gemma-2-9b-it</div>
        </div>
        <div>
          <label>Adapter ID (LoRA)</label><br/>
          <input type='text' name='adapter_id' value='{adapter_id_default}' style='width:100%'/>
          <div class='muted'>例: user/repo（空欄可）</div>
        </div>
      </div>
      <div class='row'>
        <div>
          <label>Input JSONL</label><br/>
          <input type='text' name='input_path' value='data/elyza-tasks-100-TV_0.jsonl' style='width:100%'/>
        </div>
        <div>
          <label>Output JSONL</label><br/>
          <input type='text' name='output_path' value='outputs/ui_run.jsonl' style='width:100%'/>
        </div>
      </div>
      <div class='row'>
        <div>
          <label>Use Unsloth</label>
          <select name='use_unsloth'>
            <option value='false'>false</option>
            <option value='true'>true</option>
          </select>
        </div>
        <div>
          <label>Max New Tokens</label><br/>
          <input type='number' name='max_new_tokens' value='512'/>
        </div>
      </div>
      <div style='margin-top:12px;'>
        <button id='runBtn' type='submit'>実行（非同期）</button>
        <span class='muted'>実行中はバックグラウンドで進みます。更新で一覧が反映されます。</span>
      </div>
    </form>
  </div>

  <div class='card'>
    <h3>出力一覧</h3>
    <table>
      <thead><tr><th>#</th><th>path</th><th>action</th></tr></thead>
      <tbody>
      {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=render_index([]))


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    model_id_default = os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B-Base")
    adapter_id_default = os.getenv("ADAPTER_ID", "")
    return HTMLResponse(content=render_ui(model_id_default, adapter_id_default))


@app.post("/run")
async def run(
    model_id: str = Form(...),
    adapter_id: str = Form(""),
    input_path: str = Form(...),
    output_path: str = Form(...),
    use_unsloth: str = Form("false"),
    max_new_tokens: int = Form(512),
):
    os.environ["MAX_NEW_TOKENS"] = str(max_new_tokens)
    use_unsloth_flag = use_unsloth.lower() in ("1", "true", "yes")

    # 実運用では非同期ジョブキューに流すのが望ましいが、ここでは簡易に即時実行
    try:
        run_inference(
            model_id=model_id,
            input_jsonl=input_path,
            output_jsonl=output_path,
            adapter_id=(adapter_id or None),
            use_unsloth=use_unsloth_flag,
        )
    except Exception as e:
        # 失敗してもUIに戻す
        print("[run] failed:", e)
    return RedirectResponse(url="/ui", status_code=303)


@app.get("/view", response_class=HTMLResponse)
async def view(f: str):
    # 簡易ビューア（最初の50行）
    path = Path(f)
    if not path.is_absolute():
        path = BASE_DIR / f
    if not path.exists():
        return HTMLResponse(content=f"<pre>Not found: {f}</pre>")

    lines = []
    try:
        with open(path, "r", encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                if i >= 50:
                    break
                lines.append(line.rstrip())
    except Exception as e:
        lines = [str(e)]
    body = "\n".join(lines)
    return HTMLResponse(content=f"<pre>{body}</pre>")


# ---- JSON API (for TS frontend) ----

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/outputs")
async def api_outputs():
    rows = []
    for p in sorted(OUTPUTS_DIR.glob("*.jsonl")):
        rows.append({
            "path": str(p.relative_to(BASE_DIR)),
            "size": p.stat().st_size,
            "mtime": int(p.stat().st_mtime),
        })
    return {"items": rows}


@app.get("/api/view")
async def api_view(f: str, max_lines: int = 200):
    path = Path(f)
    if not path.is_absolute():
        path = BASE_DIR / f
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    lines = []
    with open(path, "r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i >= max_lines:
                break
            lines.append(line.rstrip("\n"))
    return {"path": str(path), "lines": lines}


@app.post("/api/run")
async def api_run(req: Request):
    body = await req.json()
    model_id = body.get("model_id") or os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B-Base")
    adapter_id = body.get("adapter_id") or None
    input_path = body.get("input_path")
    output_path = body.get("output_path")
    use_unsloth = bool(body.get("use_unsloth", False))
    max_new_tokens = int(body.get("max_new_tokens", 512))

    if not input_path or not output_path:
        return JSONResponse(status_code=400, content={"error": "input_path and output_path are required"})

    os.environ["MAX_NEW_TOKENS"] = str(max_new_tokens)
    try:
        run_inference(
            model_id=model_id,
            input_jsonl=input_path,
            output_jsonl=output_path,
            adapter_id=adapter_id,
            use_unsloth=use_unsloth,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"status": "started", "output": output_path}


