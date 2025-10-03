from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
import uuid
import subprocess
import socket
import os
import json
import time
from dotenv import load_dotenv

from src.infer import run_inference
from src.eval_gemini import evaluate_jsonl, load_eval_config_from_env, NonRetryableError, verify_api_key
from src.utils.jsonl_io import write_jsonl
from src.eval_runner import (
    create_job,
    start_job,
    get_job_status,
    cancel_job,
    list_jobs,
    dry_run_estimate,
)


app = FastAPI(title="LLM Experiments Dashboard")

# CORS (for Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5174"))

# ---- Logging ----
logger = logging.getLogger("llm_app")
if not logger.handlers:
    _handler = RotatingFileHandler(
        LOGS_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Serve built frontend if available (no need for Vite dev server)
if FRONTEND_DIST_DIR.exists():
    app.mount("/fe", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
    @app.get("/fe")
    async def fe_root():
        return RedirectResponse(url="/fe/")


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


def _start_vite_if_needed() -> None:
    try:
        if _is_port_open("127.0.0.1", FRONTEND_PORT):
            logger.info(f"vite already listening on {FRONTEND_PORT}")
            return
        if not (FRONTEND_DIR / "package.json").exists():
            logger.info("frontend directory not found; skip vite start")
            return
        cmd = (
            f"cd '{FRONTEND_DIR}' && (pkill -f vite || true); "
            f"npm ci --no-audit --no-fund && "
            f"VITE_API_BASE=http://localhost:8000 nohup npm run dev -- --host 0.0.0.0 --port {FRONTEND_PORT} > ../vite.out 2>&1 &"
        )
        subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"vite starting on port {FRONTEND_PORT}")
    except Exception as e:
        logger.exception(f"failed to start vite: {e}")


@app.on_event("startup")
async def _on_startup():
    # Auto start frontend dev server unless disabled
    auto = os.getenv("AUTO_START_FRONTEND", "true").lower() in ("1", "true", "yes")
    if auto:
        _start_vite_if_needed()


@app.get("/admin/start_frontend")
async def admin_start_frontend():
    _start_vite_if_needed()
    return {"status": "starting", "port": FRONTEND_PORT}


@app.api_route("/admin/restart_backend", methods=["GET", "POST"])
async def admin_restart_backend():
    try:
        cmd = f"cd '{BASE_DIR}' && bash scripts/restart_backend.sh"
        subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("admin requested backend restart")
        return {"status": "restarting"}
    except Exception as e:
        error_id = uuid.uuid4().hex
        logger.exception(f"[admin_restart_backend] error_id={error_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "internal_error"}, headers={"X-Error-Id": error_id})

# Load .env for API keys (e.g., GEMINI_API_KEY)
load_dotenv()


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
    <h3>Gemini API Key</h3>
    <div class='row'>
      <div>
        <div class='muted'>現在の状態: <span id='geminiState'>チェック中...</span></div>
      </div>
    </div>
    <form method='post' action='/api/gemini_key' onsubmit="setTimeout(()=>{{document.getElementById('geminiState').textContent='送信しました';}},0)">
      <div class='row'>
        <div>
          <input type='password' name='api_key' placeholder='Paste API Key' style='width:360px' />
        </div>
        <div>
          <button type='submit'>保存</button>
        </div>
      </div>
    </form>
  </div>

  <div class='card'>
    <h3>評価用ファイルのアップロード（.jsonl / .txt / .json）</h3>
    <form method='post' action='/api/eval_upload' enctype='multipart/form-data' target='_blank'>
      <input type='file' name='file' accept='.jsonl,.txt,.json' />
      <button type='submit'>Geminiで採点</button>
      <span class='muted'>結果JSONは新しいタブに表示されます。</span>
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
        # 失敗詳細はログにのみ出力し、UIには出さない
        error_id = uuid.uuid4().hex
        logger.exception(f"[run] failed error_id={error_id}: {e}")
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
        error_id = uuid.uuid4().hex
        logger.exception(f"[api_run] error_id={error_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "internal_error"}, headers={"X-Error-Id": error_id})
    return {"status": "started", "output": output_path}


# ---- File upload + Gemini evaluation ----

@app.post("/api/eval_upload")
async def api_eval_upload(file: UploadFile = File(...), max_records: int = 3):
    """互換性のため残置。内部的には新しいeval_startを使用（簡易版：ドライラン→即実行→完了待ち）"""
    try:
        # 新しいAPI経由で実行（ドライラン→本番）
        # まずドライラン
        file_content = await file.read()
        await file.seek(0)  # ファイルポインタをリセット
        
        # eval_start でドライラン
        ts = int(time.time())
        raw_name = f"{ts}_{file.filename}"
        saved_path = UPLOADS_DIR / raw_name
        with open(saved_path, "wb") as f:
            f.write(file_content)

        # 正規化
        in_jsonl_path = saved_path
        if saved_path.suffix.lower() in {".txt", ".log"}:
            jsonl_path = UPLOADS_DIR / f"{saved_path.stem}.jsonl"
            rows = []
            with open(saved_path, "r", encoding="utf-8", errors="ignore") as rf:
                for i, line in enumerate(rf):
                    text = line.strip()
                    if not text:
                        continue
                    rows.append({"task_id": i + 1, "input": "", "output": text})
            write_jsonl(str(jsonl_path), rows)
            in_jsonl_path = jsonl_path
        elif saved_path.suffix.lower() == ".json":
            jsonl_path = UPLOADS_DIR / f"{saved_path.stem}.jsonl"
            try:
                with open(saved_path, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
            except Exception as e:
                error_id = uuid.uuid4().hex
                logger.exception(f"[api_eval_upload] invalid json error_id={error_id}: {e}")
                return JSONResponse(status_code=400, content={"error": "invalid_json", "error_id": error_id})
            rows: list[dict] = []
            if isinstance(data, list):
                for i, obj in enumerate(data):
                    if isinstance(obj, dict):
                        rows.append(obj)
                    else:
                        rows.append({"task_id": i + 1, "input": "", "output": str(obj)})
            elif isinstance(data, dict):
                rows.append(data)
            else:
                rows.append({"task_id": 1, "input": "", "output": str(data)})
            write_jsonl(str(jsonl_path), rows)
            in_jsonl_path = jsonl_path

        # ドライラン見積（省略可能だが、ここでは従来動作に近づけるため実行）
        cfg = load_eval_config_from_env()
        estimate = dry_run_estimate(str(in_jsonl_path), cfg)
        if estimate.get("error"):
            logger.warning(f"[api_eval_upload] dry_run failed: {estimate['error']}")

        # ジョブ作成→開始（max_recordsは無視し、全件採点）
        run_id = create_job(
            input_path=str(in_jsonl_path),
            qps=None,
            prompt_version=None,
            cache=None,
        )
        start_job(run_id)

        # 完了を待つ（簡易的にポーリング、最大10分）
        max_wait = 600  # 10分
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = get_job_status(run_id)
            if not status:
                break
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)

        # 結果を取得
        status = get_job_status(run_id)
        if not status:
            return JSONResponse(status_code=500, content={"error": "job_not_found"})

        if status["status"] == "failed":
            return JSONResponse(status_code=500, content={"error": "evaluation_failed", "details": status.get("error")})

        # 結果ファイルから要約を作成
        out_path = Path(status["output_path"])
        num = 0
        total = 0.0
        head: list[dict] = []
        try:
            with open(out_path, "r", encoding="utf-8") as rf:
                for i, line in enumerate(rf):
                    obj = json.loads(line)
                    num += 1
                    try:
                        total += float(obj.get("score", 0))
                    except Exception:
                        pass
                    if i < 5:
                        head.append(obj)
        except Exception:
            pass

        avg = (total / num) if num else 0.0
        return {"status": "ok", "output": str(out_path.relative_to(BASE_DIR)), "count": num, "avg_score": avg, "head": head}
    except Exception as e:
        error_id = uuid.uuid4().hex
        logger.exception(f"[api_eval_upload] unexpected error_id={error_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "internal_error"}, headers={"X-Error-Id": error_id})


# ---- Gemini status ----

@app.get("/api/gemini_status")
async def gemini_status():
    key = os.getenv("GEMINI_API_KEY", "")
    configured = bool(key)
    masked = "\u25CF" * 5 if configured else ""
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    # オプション: 即時検証（軽量）
    ok, reason = verify_api_key(load_eval_config_from_env()) if configured else (False, "not_set")
    return {"configured": configured, "valid": ok, "reason": (reason if not ok else ""), "masked": masked, "model": model}


@app.post("/api/gemini_key")
async def set_gemini_key(req: Request):
    # Accept JSON, form, or raw text. Do not leak details to client.
    api_key = None
    try:
        body = await req.json()
        if isinstance(body, dict):
            api_key = body.get("api_key")
    except Exception:
        body = None
    if not api_key:
        try:
            form = await req.form()
            api_key = form.get("api_key") if form else None
        except Exception:
            pass
    if not api_key:
        try:
            raw = await req.body()
            text = (raw or b"").decode("utf-8", errors="ignore").strip()
            # accept non-empty raw as key
            if text:
                api_key = text
        except Exception:
            pass
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "api_key is required"})
    # 形式: 簡易バリデーション（1文字などの明らかな誤りをはじく）
    api_key = str(api_key).strip()
    if len(api_key) < 10:
        return JSONResponse(status_code=400, content={"error": "invalid_api_key", "reason": "too_short"})

    # 一時設定して有効性を検証
    prev = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = api_key
    ok, reason = verify_api_key(load_eval_config_from_env())

    # 永続化: プロジェクト直下の .env に保存/更新（有効/無効に関わらず保持して状態可視化を優先）
    try:
        env_path = BASE_DIR / ".env"
        lines: list[str] = []
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8", errors="ignore") as fp:
                lines = fp.readlines()
        updated = False
        new_lines = []
        for ln in lines:
            if ln.strip().startswith("GEMINI_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={api_key}\n")
                updated = True
            else:
                new_lines.append(ln)
        if not updated:
            new_lines.append(f"GEMINI_API_KEY={api_key}\n")
        with open(env_path, "w", encoding="utf-8") as fp:
            fp.writelines(new_lines)
    except Exception:
        # 失敗してもプロセス内には保持する（ログには鍵は出さない）
        pass

    # 有効性結果をそのまま返す（無効でも200で反映し、ステータスは valid=false）
    return {"status": "ok", "masked": "\u25CF" * 5, "valid": bool(ok), "reason": (reason or "")}


# ---- Log utilities ----

def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            lines = fp.readlines()
        return [ln.rstrip("\n") for ln in lines[-max(0, n):]]
    except Exception as e:
        logger.exception(f"[logs_tail] failed to read {path}: {e}")
        return [f"<read error: {e}>"]


@app.get("/api/logs_tail")
async def api_logs_tail(file: str = "app", n: int = 200):
    files = {
        "app": LOGS_DIR / "app.log",
        "uvicorn": BASE_DIR / "uvicorn.out",
        "vite": BASE_DIR / "vite.out",
    }
    key = (file or "app").lower()
    path = files.get(key)
    if not path:
        return JSONResponse(status_code=400, content={"error": "invalid_file"})
    if not path.exists():
        return {"file": key, "path": str(path.relative_to(BASE_DIR)), "lines": []}
    lines = _tail_lines(path, int(n or 200))
    return {"file": key, "path": str(path.relative_to(BASE_DIR)), "lines": lines}


@app.get("/api/error_lookup")
async def api_error_lookup(error_id: str, file: str = "app", context: int = 3, max_hits: int = 20):
    files = {
        "app": LOGS_DIR / "app.log",
        "uvicorn": BASE_DIR / "uvicorn.out",
        "vite": BASE_DIR / "vite.out",
    }
    key = (file or "app").lower()
    path = files.get(key)
    if not error_id:
        return JSONResponse(status_code=400, content={"error": "error_id_required"})
    if not path or not path.exists():
        return JSONResponse(status_code=404, content={"error": "log_not_found"})

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            lines = [ln.rstrip("\n") for ln in fp.readlines()]
    except Exception as e:
        logger.exception(f"[error_lookup] failed to read {path}: {e}")
        return JSONResponse(status_code=500, content={"error": "read_failed"})

    ctx = max(0, int(context))
    hits: list[dict] = []
    for idx, ln in enumerate(lines):
        if error_id in ln:
            start = max(0, idx - ctx)
            end = min(len(lines), idx + ctx + 1)
            hits.append({
                "line": idx + 1,
                "snippet": lines[start:end],
            })
            if len(hits) >= max(1, int(max_hits)):
                break
    return {"file": key, "path": str(path.relative_to(BASE_DIR)), "error_id": error_id, "hits": hits}


# ---- 非同期評価ラン API ----

@app.post("/api/eval_start")
async def api_eval_start(
    file: UploadFile = File(...),
    qps: Optional[float] = Form(None),
    prompt_version: Optional[str] = Form(None),
    cache: Optional[str] = Form(None),
    dry_run: bool = Form(False),
):
    """評価ランを開始（ドライランまたは本番実行）"""
    try:
        # ファイル保存
        ts = int(time.time())
        raw_name = f"{ts}_{file.filename}"
        saved_path = UPLOADS_DIR / raw_name
        with open(saved_path, "wb") as f:
            f.write(await file.read())

        # 正規化（.txt/.jsonの場合）
        in_jsonl_path = saved_path
        if saved_path.suffix.lower() in {".txt", ".log"}:
            jsonl_path = UPLOADS_DIR / f"{saved_path.stem}.jsonl"
            rows = []
            with open(saved_path, "r", encoding="utf-8", errors="ignore") as rf:
                for i, line in enumerate(rf):
                    text = line.strip()
                    if not text:
                        continue
                    rows.append({"task_id": i + 1, "input": "", "output": text})
            write_jsonl(str(jsonl_path), rows)
            in_jsonl_path = jsonl_path
        elif saved_path.suffix.lower() == ".json":
            jsonl_path = UPLOADS_DIR / f"{saved_path.stem}.jsonl"
            try:
                with open(saved_path, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
            except Exception as e:
                error_id = uuid.uuid4().hex
                logger.exception(f"[api_eval_start] invalid json error_id={error_id}: {e}")
                return JSONResponse(status_code=400, content={"error": "invalid_json", "error_id": error_id})
            rows: list[dict] = []
            if isinstance(data, list):
                for i, obj in enumerate(data):
                    if isinstance(obj, dict):
                        rows.append(obj)
                    else:
                        rows.append({"task_id": i + 1, "input": "", "output": str(obj)})
            elif isinstance(data, dict):
                rows.append(data)
            else:
                rows.append({"task_id": 1, "input": "", "output": str(data)})
            write_jsonl(str(jsonl_path), rows)
            in_jsonl_path = jsonl_path

        # ドライラン
        if dry_run:
            cfg = load_eval_config_from_env()
            if qps is not None:
                cfg.qps = qps
            if prompt_version is not None:
                cfg.prompt_version = prompt_version
            estimate = dry_run_estimate(str(in_jsonl_path), cfg)
            return {"status": "dry_run", "estimate": estimate}

        # 本番: ジョブ作成→開始
        run_id = create_job(
            input_path=str(in_jsonl_path),
            qps=qps,
            prompt_version=prompt_version,
            cache=cache,
        )
        start_job(run_id)

        return {"status": "started", "run_id": run_id}

    except Exception as e:
        error_id = uuid.uuid4().hex
        logger.exception(f"[api_eval_start] error_id={error_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "internal_error"}, headers={"X-Error-Id": error_id})


@app.get("/api/eval_status")
async def api_eval_status(run_id: str):
    """評価ランの進捗を取得"""
    status = get_job_status(run_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "run_not_found"})
    return status


@app.post("/api/eval_cancel")
async def api_eval_cancel(run_id: str = Form(...)):
    """評価ランをキャンセル"""
    success = cancel_job(run_id)
    if not success:
        return JSONResponse(status_code=400, content={"error": "cannot_cancel"})
    return {"status": "cancelled", "run_id": run_id}


@app.get("/api/eval_runs")
async def api_eval_runs(limit: int = 50):
    """評価ラン一覧を取得"""
    runs = list_jobs(limit=limit)
    return {"runs": runs}


@app.get("/api/eval_result")
async def api_eval_result(run_id: str, format: str = "jsonl"):
    """評価結果をダウンロード"""
    from fastapi.responses import FileResponse
    
    status = get_job_status(run_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "run_not_found"})
    
    if format == "csv":
        path = Path(status.get("output_path", "")).with_suffix(".summary.csv")
    else:
        path = OUTPUTS_DIR / f"gemini_eval_{run_id}.jsonl"
    
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "file_not_found"})
    
    return FileResponse(path, filename=path.name)
