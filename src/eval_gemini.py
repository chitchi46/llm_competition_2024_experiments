import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import google.generativeai as genai
except Exception as e:  # pragma: no cover
    genai = None

from src.utils.jsonl_io import iter_jsonl, write_jsonl


@dataclass
class EvalConfig:
    model: str
    qps: float
    max_retries: int
    prompt_version: str
    cache_dir: Optional[str]


def load_eval_config_from_env() -> EvalConfig:
    return EvalConfig(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        qps=float(os.getenv("GEMINI_QPS", "1.0")),
        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "5")),
        prompt_version=os.getenv("GEMINI_PROMPT_VERSION", "v1"),
        cache_dir=os.getenv("GEMINI_CACHE_DIR"),
    )


def build_prompt(input_text: str, output_text: str, reference_text: Optional[str]) -> str:
    rubric = (
        "あなたは日本語の回答を採点する評価者です。1から5の整数で採点します。\n"
        "評価観点:\n"
        "- 指示遵守\n- 正確性\n- 情報の網羅性\n- 日本語表現\n"
        "総合的に判断して1(悪い)〜5(優れている)の整数を出し、短い根拠も添えてください。\n"
        "出力はJSONのみ: {\"score\": int(1-5), \"reason\": str, \"suggestion\": str}。他のテキストは出さない。"
    )
    example = (
        f"# 入力\n{input_text}\n\n"
        f"# 生成出力\n{output_text}\n\n"
    )
    if reference_text:
        example += f"# 参考\n{reference_text}\n\n"
    return rubric + "\n\n" + example


def _cache_key(task_id: str, output_text: str, prompt_version: str) -> str:
    h = hashlib.sha256((task_id + "::" + output_text + "::" + prompt_version).encode("utf-8")).hexdigest()
    return h


def _maybe_read_cache(cache_dir: Optional[str], key: str) -> Optional[Dict]:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _write_cache(cache_dir: Optional[str], key: str, obj: Dict) -> None:
    if not cache_dir:
        return
    path = Path(cache_dir) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class RateLimiter:
    def __init__(self, qps: float):
        self.min_interval = 1.0 / max(1e-6, qps)
        self.last: float = 0.0

    def wait(self):
        now = time.time()
        sleep_for = self.last + self.min_interval - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self.last = time.time()


class NonRetryableError(Exception):
    pass


def _normalize_gemini_text(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()
        # 先頭行に言語名が付いている場合を除去
        parts = s.split("\n", 1)
        if len(parts) == 2 and parts[0] and not parts[0].lstrip().startswith("{"):
            s = parts[1]
    # JSON以外の前置き/後置きを粗く除去
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last+1]
    return s.strip()


def _call_gemini(prompt: str, cfg: EvalConfig) -> Dict:
    if genai is None:
        raise RuntimeError("google-generativeai がインストールされていません")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise NonRetryableError("GEMINI_API_KEY が未設定です")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg.model)

    response = model.generate_content(prompt)
    # ブロック理由を確認
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        import logging
        logging.warning(f"[Gemini] prompt_feedback={response.prompt_feedback}")
        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            raise ValueError(f"Gemini blocked prompt. Reason: {response.prompt_feedback.block_reason}")
    raw_text = response.text if hasattr(response, 'text') else ""
    text = _normalize_gemini_text(raw_text)
    # 期待形式: JSON のみ
    try:
        if not text.strip():
            # 空応答の場合はエラーログを出してリトライ対象にする
            raise ValueError(f"Empty response from Gemini. Raw: {raw_text[:200]}")
        obj = json.loads(text)
        if not isinstance(obj, dict) or "score" not in obj:
            raise ValueError(f"Invalid JSON shape (missing 'score'). Parsed: {obj}")
        return obj
    except Exception as e:
        # ログに詳細を残してリトライ
        import logging
        logging.error(f"[Gemini parse error] raw={raw_text[:300]} normalized={text[:300]} error={e}")
        raise e


def verify_api_key(cfg: EvalConfig) -> tuple[bool, str]:
    """API Key の有効性を簡易検証する。

    - key未設定や認証エラーなどを検出
    - モデルの `count_tokens` を用い、生成を行わず低コストで確認
    """
    if genai is None:
        return False, "library_not_installed"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False, "not_set"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(cfg.model)
        # tokenカウントで軽量チェック
        _ = model.count_tokens("ping")
        return True, ""
    except Exception as e:  # 認証/権限/レート等を包括
        msg = str(e)
        if len(msg) > 300:
            msg = msg[:300]
        return False, msg


@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(int(os.getenv("GEMINI_MAX_RETRIES", "3"))),
)
def call_gemini_with_retry(prompt: str, cfg: EvalConfig) -> Dict:
    return _call_gemini(prompt, cfg)


def evaluate_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    cfg: EvalConfig,
    max_records: Optional[int] = None,
) -> None:
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(cfg.qps)
    results: List[Dict] = []

    for i, rec in enumerate(iter_jsonl(input_jsonl)):
        if max_records is not None and i >= max_records:
            break

        task_id = str(rec.get("task_id"))
        input_text = rec.get("input") or rec.get("instruction") or ""
        output_text = rec.get("output") or ""
        reference_text = rec.get("reference")

        key = _cache_key(task_id, output_text, cfg.prompt_version)
        cached = _maybe_read_cache(cfg.cache_dir, key)
        if cached is not None:
            obj = cached
        else:
            prompt = build_prompt(input_text, output_text, reference_text)
            limiter.wait()
            obj = call_gemini_with_retry(prompt, cfg)
            _write_cache(cfg.cache_dir, key, obj)

        score = int(obj.get("score", 0))
        reason = str(obj.get("reason", ""))
        suggestion = str(obj.get("suggestion", ""))

        results.append({
            "task_id": task_id,
            "score": score,
            "reason": reason,
            "suggestion": suggestion,
            "prompt_version": cfg.prompt_version,
            "model_name": cfg.model,
        })

    write_jsonl(output_jsonl, results)

    # 集計の簡易出力（CSV）
    try:
        df = pd.DataFrame(results)
        if not df.empty:
            agg_path = str(Path(output_jsonl).with_suffix(".summary.csv"))
            summary = df["score"].describe()
            summary.to_csv(agg_path)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate JSONL using Gemini 2.5 Flash")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--cache", default=os.getenv("GEMINI_CACHE_DIR"))
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    p.add_argument("--qps", type=float, default=float(os.getenv("GEMINI_QPS", "1.0")))
    p.add_argument("--prompt-version", default=os.getenv("GEMINI_PROMPT_VERSION", "v1"))
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_eval_config_from_env()
    cfg.model = args.model
    cfg.qps = args.qps
    cfg.prompt_version = args.prompt_version
    cfg.cache_dir = args.cache

    evaluate_jsonl(
        input_jsonl=args.input,
        output_jsonl=args.output,
        cfg=cfg,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()


