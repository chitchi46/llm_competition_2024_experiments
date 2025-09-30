## Gemini 評価 開発ドキュメント（Linux/WSL 専用）

本ドキュメントは、READMEから分離した「Gemini API を用いた評価（開発者向け）」の設計と運用手順をまとめたものです。Windows/PowerShell は対象外です。

---

## 目的とスコープ
- ELYZA-tasks-100-TV などの出力を Gemini 2.5 により自動評価するための再現性ある評価基盤を提供
- CLI と Web UI の両系統から同一パイプラインを呼び出す一貫性
- 安定性（QPS 制御 / リトライ / キャッシュ）と運用性（ログ / サマリ）を確保

非対象: Windows/PowerShell 対応、外部DB以外の永続化、複雑な多指標化（本フェーズは単一スコア）

---

## 構成要素
- エントリポイント: `src/eval_gemini.py`
  - 引数: `--input`, `--output`, `--max-records`, `--cache`, `--model`, `--qps`, `--prompt-version`
  - 機能: QPS 制御、指数バックオフ再試行、オンディスクキャッシュ、集計 CSV 出力
- 実行スクリプト: `scripts/run_eval_gemini.sh`
- 環境変数: `.env`（`env.example` をコピー）
- Web 経由: FastAPI `POST /api/eval_upload`（.jsonl/.json/.txt を自動正規化→評価→要約）
- ログ: `logs/app.log`

---

## セットアップ（Linux/WSL）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

`.env` の主要項目:
- `GEMINI_API_KEY`（必須）
- `GEMINI_MODEL`（既定: gemini-2.5-flash）
- `GEMINI_QPS`（既定: 1.0）
- `GEMINI_MAX_RETRIES`（既定: 5）
- `GEMINI_PROMPT_VERSION`（既定: v1）
- 任意: `GEMINI_CACHE_DIR`（オンディスクキャッシュを有効化する場合に指定）

---

## まず検証（エラーなく適切に評価できるか）
最短・確実な順に検証します。各ステップが成功したら次へ進みます。

1) 依存関係と鍵
```bash
source .venv/bin/activate
pip install -r requirements.txt
cp -n env.example .env || true
export GEMINI_API_KEY="<実キー>"
python - <<'PY'
from dotenv import load_dotenv; load_dotenv();
from src.eval_gemini import load_eval_config_from_env, verify_api_key
ok, reason = verify_api_key(load_eval_config_from_env())
print({"ok": ok, "reason": reason})
PY
```
期待値: `{"ok": True, "reason": ""}`（Falseの場合は鍵や権限を再確認）

2) 最小サンプルで実行（5件）
```bash
mkdir -p eval
python -m src.eval_gemini --input data/elyza-tasks-100-TV_0.jsonl \
  --output eval/smoke_eval.jsonl --max-records 5 --qps 0.5 --prompt-version v1
```
期待値:
- `eval/smoke_eval.jsonl` が5行で生成
- `eval/smoke_eval.summary.csv` が生成され、scoreの記述統計が出力

3) 既存出力（uploads）を評価
```bash
ls -1 uploads/*.jsonl | head -n 1 | while read f; do \
  python -m src.eval_gemini --input "$f" --output eval/upload_eval.jsonl --max-records 5; done
```
期待値: 実行が完了し、`eval/upload_eval.jsonl` が生成

4) キャッシュ確認（同条件再実行）
```bash
export GEMINI_CACHE_DIR=.cache/gemini
python -m src.eval_gemini --input data/elyza-tasks-100-TV_0.jsonl \
  --output eval/smoke_eval_cached.jsonl --max-records 5 --prompt-version v1
```
期待値: 2回目はAPI呼び出しが大幅減（速度向上）。内容は同等。

上記がすべて成功したら、UI経由（/api/eval_upload）でファイルを投げて同様の結果が得られることを確認してください。

---

## CLI 実行
```bash
# 例: 100件だけ評価
export GEMINI_API_KEY=...  # 実キーを設定
bash scripts/run_eval_gemini.sh outputs/qwen3_out.jsonl eval/gemini_eval.jsonl 100

# 直接モジュール実行
python -m src.eval_gemini --input outputs/qwen3_out.jsonl --output eval/gemini_eval.jsonl --max-records 100 \
  --model gemini-2.5-flash --qps 1.0 --prompt-version v1 --cache .cache/gemini
```

出力:
- `eval/gemini_eval.jsonl`: `{task_id, score(1-5), reason, suggestion, prompt_version, model_name}`
- `eval/gemini_eval.summary.csv`: スコアの基本統計（参考）

キャッシュ:
- `GEMINI_CACHE_DIR` または `--cache` 指定時、出力テキストとプロンプト版に基づくキーで JSON を保存
- 同一条件で再実行するとキャッシュ命中→API 呼び出しが抑制される

---

## Web 経由（開発用UI）
- エンドポイント: `POST /api/eval_upload`
  - 受理: `.jsonl / .json / .txt`
  - `.json` は配列/単一オブジェクトを JSONL に正規化
  - `.txt` は非空行ごとに 1 レコード化
  - 既定では最大 50 件を評価（`max_records` で変更可能）
- レスポンス: `{"status":"ok","output":"outputs/...jsonl","count":N,"avg_score":x.xx,"head":[...]}`

補助 API:
- `GET /api/gemini_status`: 鍵の設定・有効性の軽量チェック
- `POST /api/gemini_key`: 鍵の保存（`.env` を更新／失敗時はロールバック）

---

## 実装の要点
- QPS 制御: 単純な最小間隔スリープで安定化
- リトライ: `tenacity` による指数バックオフ（最大試行は環境変数で制御）
- 正規化: Gemini 応答のコードフェンス/前後語を除去し JSON のみに整形
- スキーマ: `{"score": int(1-5), "reason": str, "suggestion": str}`
- 集計: JSONL のあとに `pandas` で基本統計を CSV 出力

---

## トラブルシュート
- 401/permission: `GEMINI_API_KEY` の権限・入力ミスを確認
- JSON 解析失敗: モデルの余分なテキストを除去できているか確認（プロンプト版を固定）
- レート制限: `GEMINI_QPS` を下げる、`GEMINI_MAX_RETRIES` を増やす
- キャッシュ不一致: `GEMINI_PROMPT_VERSION` を変更したか、出力テキストが変更されていないか確認

---

## 将来拡張（Convex 永続化）
- 外部DB: Convex を採用。評価ラン（Run）のメタデータ/サマリを保存し、過去ラン比較を可能にする
- 追加環境変数（案）: `CONVEX_URL`, `CONVEX_TOKEN`
- データモデル（案）:
  - `eval_runs`: `run_id`, `created_at`, `dataset_id`, `input_hash`, `model_name`, `prompt_version`, `qps`, `max_records`, `avg_score`, `score_hist`, `output_path`, `summary_csv_path`, `samples`
- フロー（案）:
  1) 評価完了後に Convex へ保存（失敗時はログのみ・評価は成功とする）
  2) `GET /api/eval_runs` で一覧、`GET /api/eval_compare?run_a&run_b` で比較（平均/分布/抜粋）

---

## 開発メモ
- コード: `src/eval_gemini.py`, `src/app.py`, `scripts/run_eval_gemini.sh`
- 既存 UI はキー登録とファイルアップロード→評価→結果パス提示まで対応済み


