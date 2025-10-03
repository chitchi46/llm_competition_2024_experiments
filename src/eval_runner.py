"""非同期評価ジョブの実行・管理"""
import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

from src.eval_gemini import (
    EvalConfig,
    evaluate_jsonl,
    load_eval_config_from_env,
    NonRetryableError,
)
from src.utils.jsonl_io import iter_jsonl

logger = logging.getLogger("llm_app.eval_runner")

RUNS_DIR = Path(__file__).resolve().parent.parent / "outputs" / ".runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvalJobInfo:
    run_id: str
    input_path: str
    output_path: str
    status: str  # pending, running, completed, failed, cancelled
    total: int
    done: int
    failed: int
    cached_hits: int
    avg_score: float
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    elapsed: float
    eta: float
    config: Dict  # qps, prompt_version, model, cache_dir
    error: Optional[str]


# グローバルなジョブレジストリ（run_id → EvalJobInfo）
_jobs: Dict[str, EvalJobInfo] = {}
_jobs_lock = threading.Lock()
_cancel_flags: Set[str] = set()


def _load_job_from_disk(run_id: str) -> Optional[EvalJobInfo]:
    """ディスクからジョブ情報を読み込み"""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvalJobInfo(**data)
    except Exception as e:
        logger.exception(f"Failed to load job {run_id}: {e}")
        return None


def _save_job_to_disk(job: EvalJobInfo) -> None:
    """ジョブ情報をディスクに保存"""
    path = RUNS_DIR / f"{job.run_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(job), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Failed to save job {job.run_id}: {e}")


def _count_existing_results(output_path: str) -> int:
    """既存の出力ファイルから処理済み件数をカウント"""
    if not Path(output_path).exists():
        return 0
    count = 0
    try:
        for _ in iter_jsonl(output_path):
            count += 1
    except Exception:
        pass
    return count


def _get_processed_task_ids(output_path: str) -> Set[str]:
    """既存の出力ファイルから処理済みtask_idの集合を取得"""
    if not Path(output_path).exists():
        return set()
    task_ids = set()
    try:
        for rec in iter_jsonl(output_path):
            task_ids.add(str(rec.get("task_id", "")))
    except Exception:
        pass
    return task_ids


def dry_run_estimate(input_path: str, cfg: EvalConfig) -> Dict:
    """ドライラン: 所要時間とコストを見積もり"""
    try:
        import google.generativeai as genai
    except Exception:
        return {"error": "google-generativeai not installed"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}

    # 入力件数をカウント
    total = 0
    sample_texts = []
    try:
        for i, rec in enumerate(iter_jsonl(input_path)):
            total += 1
            if i < 3:  # 最初の3件をサンプルに
                sample_texts.append(rec.get("output", ""))
    except Exception as e:
        return {"error": f"Failed to read input: {e}"}

    if total == 0:
        return {"error": "No records in input"}

    # サンプルからトークン数を見積もり
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(cfg.model)
        token_counts = []
        for text in sample_texts:
            prompt = f"evaluate: {text[:500]}"  # 簡易プロンプト
            result = model.count_tokens(prompt)
            token_counts.append(result.total_tokens)
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 100
    except Exception as e:
        logger.warning(f"Token counting failed, using default: {e}")
        avg_tokens = 100

    # 見積もり計算
    qps = cfg.qps
    min_interval = 1.0 / max(1e-6, qps)
    estimated_time = total * min_interval  # 秒

    # コスト見積もり（Gemini 2.5 Flash の参考価格: 入力$0.075/1M tokens, 出力$0.30/1M tokens）
    # 出力は入力の1/4と仮定
    input_tokens_total = avg_tokens * total
    output_tokens_total = input_tokens_total * 0.25
    cost_input = (input_tokens_total / 1_000_000) * 0.075
    cost_output = (output_tokens_total / 1_000_000) * 0.30
    estimated_cost = cost_input + cost_output

    return {
        "total": total,
        "avg_tokens": round(avg_tokens, 1),
        "qps": qps,
        "estimated_time_sec": round(estimated_time, 1),
        "estimated_time_min": round(estimated_time / 60, 1),
        "estimated_cost_usd": round(estimated_cost, 4),
    }


def create_job(
    input_path: str,
    qps: Optional[float] = None,
    prompt_version: Optional[str] = None,
    cache: Optional[str] = None,
) -> str:
    """新しいジョブを作成し、run_idを返す"""
    run_id = f"{int(time.time())}_{Path(input_path).stem}"
    output_path = str(Path(__file__).resolve().parent.parent / "outputs" / f"gemini_eval_{run_id}.jsonl")

    cfg = load_eval_config_from_env()
    if qps is not None:
        cfg.qps = qps
    if prompt_version is not None:
        cfg.prompt_version = prompt_version
    if cache is not None:
        cfg.cache_dir = cache if cache.lower() not in ("off", "false", "") else None

    # 入力件数をカウント
    total = 0
    try:
        for _ in iter_jsonl(input_path):
            total += 1
    except Exception as e:
        raise ValueError(f"Failed to read input: {e}")

    job = EvalJobInfo(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        status="pending",
        total=total,
        done=0,
        failed=0,
        cached_hits=0,
        avg_score=0.0,
        created_at=time.time(),
        started_at=None,
        completed_at=None,
        elapsed=0.0,
        eta=0.0,
        config={
            "qps": cfg.qps,
            "prompt_version": cfg.prompt_version,
            "model": cfg.model,
            "cache_dir": cfg.cache_dir,
        },
        error=None,
    )

    with _jobs_lock:
        _jobs[run_id] = job
    _save_job_to_disk(job)

    return run_id


def start_job(run_id: str) -> None:
    """ジョブをバックグラウンドで開始"""
    with _jobs_lock:
        job = _jobs.get(run_id)
        if not job:
            job = _load_job_from_disk(run_id)
            if not job:
                raise ValueError(f"Job {run_id} not found")
            _jobs[run_id] = job

    if job.status == "running":
        logger.warning(f"Job {run_id} is already running")
        return

    def _run():
        try:
            job.status = "running"
            job.started_at = time.time()
            _save_job_to_disk(job)

            cfg = EvalConfig(
                model=job.config["model"],
                qps=job.config["qps"],
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "5")),
                prompt_version=job.config["prompt_version"],
                cache_dir=job.config.get("cache_dir"),
            )

            # 既存の処理済みtask_idを取得
            processed_ids = _get_processed_task_ids(job.output_path)
            job.done = len(processed_ids)

            # 評価実行（逐次追記）
            from src.eval_gemini import (
                iter_jsonl,
                build_prompt,
                call_gemini_with_retry,
                RateLimiter,
                _cache_key,
                _maybe_read_cache,
                _write_cache,
            )

            limiter = RateLimiter(cfg.qps)
            results = []
            scores_sum = 0.0

            # 出力ファイルを追記モードで開く
            output_file = open(job.output_path, "a", encoding="utf-8")

            try:
                for rec in iter_jsonl(job.input_path):
                    # キャンセルチェック
                    if run_id in _cancel_flags:
                        job.status = "cancelled"
                        job.error = "Cancelled by user"
                        break

                    task_id = str(rec.get("task_id"))
                    if task_id in processed_ids:
                        continue  # スキップ

                    input_text = rec.get("input") or rec.get("instruction") or ""
                    output_text = rec.get("output") or ""
                    reference_text = rec.get("reference")

                    key = _cache_key(task_id, output_text, cfg.prompt_version)
                    cached = _maybe_read_cache(cfg.cache_dir, key)

                    if cached is not None:
                        obj = cached
                        job.cached_hits += 1
                    else:
                        prompt = build_prompt(input_text, output_text, reference_text)
                        limiter.wait()
                        obj = call_gemini_with_retry(prompt, cfg)
                        _write_cache(cfg.cache_dir, key, obj)

                    score = int(obj.get("score", 0))
                    reason = str(obj.get("reason", ""))
                    suggestion = str(obj.get("suggestion", ""))

                    result = {
                        "task_id": task_id,
                        "score": score,
                        "reason": reason,
                        "suggestion": suggestion,
                        "prompt_version": cfg.prompt_version,
                        "model_name": cfg.model,
                    }

                    # 逐次書き込み
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()

                    job.done += 1
                    scores_sum += score
                    job.avg_score = scores_sum / job.done if job.done > 0 else 0.0
                    job.elapsed = time.time() - job.started_at
                    remaining = job.total - job.done
                    job.eta = (job.elapsed / job.done * remaining) if job.done > 0 else 0.0

                    # 定期的に保存
                    if job.done % 10 == 0:
                        _save_job_to_disk(job)

            finally:
                output_file.close()

            # 完了処理
            if job.status != "cancelled":
                job.status = "completed"
            job.completed_at = time.time()
            job.elapsed = job.completed_at - job.started_at
            _save_job_to_disk(job)

            # summary.csv を生成
            try:
                import pandas as pd
                df = pd.DataFrame([json.loads(line) for line in open(job.output_path, "r", encoding="utf-8")])
                if not df.empty:
                    summary = df["score"].describe()
                    summary_path = job.output_path.replace(".jsonl", ".summary.csv")
                    summary.to_csv(summary_path)
            except Exception as e:
                logger.exception(f"Failed to generate summary: {e}")

        except NonRetryableError as e:
            job.status = "failed"
            job.error = f"Non-retryable error: {e}"
            job.completed_at = time.time()
            _save_job_to_disk(job)
            logger.exception(f"Job {run_id} failed (non-retryable): {e}")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = time.time()
            _save_job_to_disk(job)
            logger.exception(f"Job {run_id} failed: {e}")
        finally:
            if run_id in _cancel_flags:
                _cancel_flags.remove(run_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def get_job_status(run_id: str) -> Optional[Dict]:
    """ジョブの現在の状態を取得"""
    with _jobs_lock:
        job = _jobs.get(run_id)
        if not job:
            job = _load_job_from_disk(run_id)
            if not job:
                return None
            _jobs[run_id] = job

    return {
        "run_id": job.run_id,
        "status": job.status,
        "total": job.total,
        "done": job.done,
        "failed": job.failed,
        "cached_hits": job.cached_hits,
        "avg_score": round(job.avg_score, 2),
        "elapsed": round(job.elapsed, 1),
        "eta": round(job.eta, 1),
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "output_path": job.output_path,
        "error": job.error,
    }


def cancel_job(run_id: str) -> bool:
    """ジョブをキャンセル"""
    with _jobs_lock:
        job = _jobs.get(run_id)
        if not job:
            job = _load_job_from_disk(run_id)
            if not job:
                return False
            _jobs[run_id] = job

    if job.status != "running":
        return False

    _cancel_flags.add(run_id)
    return True


def list_jobs(limit: int = 50) -> List[Dict]:
    """ジョブ一覧を取得（新しい順）"""
    jobs = []
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(jobs) >= limit:
            break
        job = _load_job_from_disk(path.stem)
        if job:
            jobs.append({
                "run_id": job.run_id,
                "status": job.status,
                "total": job.total,
                "done": job.done,
                "avg_score": round(job.avg_score, 2),
                "created_at": job.created_at,
                "elapsed": round(job.elapsed, 1),
            })
    return jobs

