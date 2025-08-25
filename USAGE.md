## 使い方（セットアップ・推論・提出）

このドキュメントでは、環境構築、推論の実行方法、提出物の作り方をまとめます。コンペのルールは `README.md` を参照してください。

### 1) セットアップ
- Python 3.10+ を推奨
- 仮想環境を作成し依存関係を導入
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- 環境変数設定（`env.example` を `.env` にコピーして編集）
```bash
cp env.example .env
```

主要変数の例:
- `MODEL_ID` (例: `google/gemma-2-9b-it`)
- `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `REPETITION_PENALTY`

### 2) 推論の実行
- サンプル実行（Linux/Mac）
```bash
bash scripts/run_infer.sh
```

- サンプル実行（Windows PowerShell）
```powershell
./scripts/run_infer.ps1
```

- 任意の JSONL 入力で実行
```bash
python -m src.infer -i data/sample_inputs.jsonl -o outputs/out.jsonl -m google/gemma-2-9b-it
```

### 3) JSONL 入出力フォーマット
- 入力: `{"task_id": <int>, "input": <str>}` 形式を推奨
- 出力: `{"task_id": <int>, "input": <str>, "output": <str>}`

例（入力）:
```json
{"task_id": 0, "input": "仕事の熱意を取り戻すための100のアイデアを5つ挙げてください。"}
```

例（出力）:
```json
{"task_id": 0, "input": "仕事の熱意を取り戻すための100のアイデアを5つ挙げてください。", "output": "..."}
```

### 4) 提出物の作成
1. 推論を実行して `outputs/*.jsonl` を得る
2. モデルを Hugging Face に公開（Public）
3. `README`（モデルの使い方・推論手順）と併せて HF リポジトリにアップロード

補足:
- 秘密情報（HF トークン等）は `.env` に保存し、Git 管理外にしてください。


