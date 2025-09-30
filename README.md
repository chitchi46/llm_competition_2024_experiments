## LLM講義2024 最終課題 ― LLM開発コンペティション

本リポジトリは、松尾・岩澤研究室「大規模言語モデル講座 2024」における最終課題（LLM開発コンペティション）の成果物を管理します。以下に、コンペの目的・評価・ルール・提出物（JSON Lines 形式の例を含む）を整理します。

### 目次
- 1. コンペの目的・概要  
- 2. コンペの評価と修了要件  
- 3. ルール  
- 4. 提出物  

---

## 1. コンペの目的・概要

- **目的**:  
  ELYZA-tasks-100 の改変版 **ELYZA-tasks-100-TV** で高スコアを目指す。3週間でモデルを開発し、
  - **予選**: 自動採点によるスコア競争
  - **決勝**: 受講生による人手評価
  の2段階で競う。

- **補足（本リポの位置づけ）**:  
  本プロジェクトは「過去に実施されたコンペの再現と学習」を目的とした検証用リポジトリです。  
  当時の自動採点は Gemini 1.5 を使用していましたが、本リポでは検証に Gemini 2.5 を用います（評価傾向が一部異なる可能性があります）。

- **ベンチマーク**:  
  - **ELYZA-tasks-100**: 日本語の複雑な指示・タスクを含むベンチマーク。5段階評価。  
  - **ELYZA-tasks-100-TV**: 上記を改変し、2024年9月以降のテレビ番組内容をタスクに反映。時事性を付与し難易度を上げたもの。

---

## 2. コンペの評価と修了要件

### 評価方法（3つ）
1. **予選**: ELYZA-tasks-100-TV に対する出力を自動採点  
   - Gemini 1.5 による暫定スコア → リーダーボード反映  
   - 上位30名が決勝進出

2. **決勝**: ELYZA-tasks-100-TV に対する出力を人手評価  
   - 予選通過モデル30個を使用  
   - 受講生全員がWebフォームで評価

3. **記事投稿**:  
   - Slack専用チャンネルに投稿  
   - 他者への助言・自身の工夫を共有  
   - 上位5投稿をコントリビューション賞として表彰

---

## 3. ルール

### 受講生が行ってよいこと
- 指定モデルを用いた SFT / RLHF / DPO
- モデルの蒸留やマージ、MoE作成
- 合成データ生成（ライセンス要確認）
- Pythonライブラリを用いたRAG構築（ただし最終出力はLLMである必要あり）

### モデルの制約
- 使用可能モデル（例）:
  - LLM-jp-3 系列
  - google/gemma-2-2b, gemma-2-9b, gemma-2-27b (+ pytorch 版)
- GPU要件: **24GB (L4想定)**, CPUメモリ 48GB
- 1時間以内にタスク全体を出力可能であること

### 受講生が行ってはいけないこと
- 推論時の外部サービス利用（開発時は可）
- 開発で ELYZA-tasks-100-TV の使用
- 未改変モデルによる出力提出
- 指定外のLLM利用
- ライセンス違反
- 予選リーダーボードを用いたチューニング

---

## 4. 提出物

### 必須提出（全員）
1. タスク出力 (JSON Lines 形式)  
2. Hugging Face にアップロードしたモデルのURL (public必須)  
3. README (出力方法を記載し、モデルと共にHugging Faceにアップロード)

#### JSON Lines の例（形式の説明のみ。具体的なコードやコマンドは別ドキュメント参照）
各行に `{"task_id": <int>, "input": <str>, "output": <str>}` のレコードを記載します。

---

## 利用手順・コード類
利用方法、セットアップ、コマンド、入出力サンプルは `USAGE.md` に分離しました。

- 利用ガイド: `USAGE.md`


---

## クイックスタート

### 1) セットアップ
- Python 3.10+ を推奨
- 仮想環境を作成して依存関係をインストール
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

環境変数を設定（`env.example` を `.env` にコピーして編集）
```bash
cp env.example .env
```

### 2) サンプル推論の実行（Linux/WSL）
```bash
bash scripts/run_infer.sh
```

出力は `outputs/sample_outputs.jsonl` に保存されます。

### 3) 任意のJSONLに対して実行（Linux/WSL）
```bash
python -m src.infer -i data/sample_inputs.jsonl -o outputs/out.jsonl -m google/gemma-2-9b-it
```

本リポジトリ: [GitHub リポジトリ](https://github.com/chitchi46/llm_competition_2024_experiments)


---

## 評価（Gemini 2.5 Flash, Linux/WSL）

### 前提
- `requirements.txt` をインストール済み
- `GEMINI_API_KEY` を環境変数で設定（`.env` はコミットしない）

### 実行例
```bash
export GEMINI_API_KEY=...  # 実キーを設定
bash scripts/run_eval_gemini.sh outputs/qwen3_out.jsonl eval/gemini_eval.jsonl 100
```

### 出力
- `eval/gemini_eval.jsonl`: 行ごとに `task_id`, `score(1-5)`, `reason`, `suggestion`, `prompt_version`, `model_name`
- `eval/gemini_eval.summary.csv`: スコア分布の基本統計（参考）


---

## フロントエンド（TypeScript / React / Vite）

- 起動（別ターミナルでバックエンド `uvicorn` を 8000 番で起動済み想定）

```bash
cd frontend
npm install
npm run dev
# ブラウザで http://localhost:5174 を開く
```

- 環境変数: フロントは `VITE_API_BASE`（省略時 `http://localhost:8000`）に対応

### WSL環境でのフロントエンド起動

**問題**: WSL環境では、一部のツール（Cursor等）のターミナルがPowerShellとして実行される場合があります。

**解決方法**:

1. **一括起動スクリプトを使用（推奨）**:
```bash
wsl bash -c "cd /home/<ユーザー名>/llm_competition_2024_experiments && bash scripts/start_all.sh"
```

2. **フロントエンドのみを起動**:
```bash
wsl bash -c "cd /home/<ユーザー名>/llm_competition_2024_experiments/frontend && npm run dev -- --host 0.0.0.0 --port 5174"
```
（バックグラウンド実行させる場合は、ツールのバックグラウンドフラグを使用）

3. **アクセス方法**:
   - WSL内では: `http://localhost:5174/`
   - **Windowsブラウザからは**: `http://<WSLのIPアドレス>:5174/`
   
   WSLのIPアドレスを確認:
   ```bash
   wsl bash -c "hostname -I"
   ```
   例: `172.23.244.170` → ブラウザで `http://172.23.244.170:5174/` にアクセス

**トラブルシューティング**:
- プロセス確認: `wsl bash -c "pgrep -af vite"`
- ポート確認: `wsl bash -c "ss -tlnp | grep 5174"`
- ログ確認: `vite.out` ファイルを参照


---

## 開発ドキュメント（Gemini 評価 開発）
- 開発者向けの評価設計・運用・将来拡張（Convex連携）は分離ドキュメントに集約:
  - `docs/dev/gemini-eval-dev.md`


