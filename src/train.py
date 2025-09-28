import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset, interleave_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


def _read_json_or_jsonl_lenient(path: str):
    bad_backslash = re.compile(r"\\(?![\"\\/bfnrtu])")
    trail_comma = re.compile(r",\s*([}\]])")
    ctrl_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def _try_load(s: str):
        for step in range(3):
            try:
                return json.loads(s)
            except Exception:
                if step == 0:
                    s = trail_comma.sub(r"\1", s)
                elif step == 1:
                    s = bad_backslash.sub(r"\\\\", s)
                    s = ctrl_chars.sub(" ", s)
                else:
                    raise

    with open(path, "r", encoding="utf-8-sig") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            raw = f.read().strip()
            if not raw.lstrip().startswith("["):
                raw = "[" + raw
            if not raw.rstrip().endswith("]"):
                raw = raw + "]"
            data = _try_load(raw)
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return data["data"]
            if isinstance(data, list):
                return data
            raise ValueError("配列 JSON を解釈できませんでした。")
        rows, bad = [], 0
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(_try_load(s))
            except Exception:
                bad += 1
                if bad <= 5:
                    print(f"[warn] 壊れ行 {i} をスキップ: {s[:160]} ...")
        if bad:
            print(f"[info] 壊れ行スキップ合計: {bad}")
        return rows


def load_any(spec: str) -> Dataset:
    spec = spec.strip()
    if spec.startswith("hf:"):
        _, tail = spec.split("hf:", 1)
        if ":" not in tail:
            raise ValueError(f"'hf:repo:split' 形式で指定してください: {spec}")
        repo, split = tail.split(":", 1)
        return load_dataset(repo, split=split)
    else:
        if not os.path.exists(spec):
            raise FileNotFoundError(f"ファイルが見つかりません: {spec}")
        try:
            return load_dataset("json", data_files=spec, split="train")
        except Exception:
            data = _read_json_or_jsonl_lenient(spec)
            return Dataset.from_list(data)


def normalize_role(role: str) -> str:
    r = (role or "").lower()
    if r in ["human", "instruction", "prompt", "query", "question", "user"]:
        return "user"
    if r in ["assistant", "bot", "gpt", "answer", "response"]:
        return "assistant"
    if r == "system":
        return "system"
    return "user"


def to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    if "messages" in example:
        msgs = [
            {
                "role": normalize_role(m.get("role") or m.get("from")),
                "content": str(
                    m.get("content") or m.get("value") or m.get("text") or ""
                ),
            }
            for m in example["messages"]
        ]
    elif "instruction" in example or ("prompt" in example and "response" in example):
        system_text = example.get("system") or ""
        user_text = (
            example.get("instruction")
            or example.get("prompt")
            or example.get("question")
            or ""
        )
        input_text = example.get("input") or ""
        output_text = (
            example.get("output") or example.get("response") or example.get("answer") or ""
        )
        if input_text:
            user_text = f"{user_text}\n\n入力:\n{input_text}"
        msgs = []
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        msgs.append({"role": "user", "content": user_text})
        if output_text:
            msgs.append({"role": "assistant", "content": output_text})
    elif "text" in example:
        msgs = [{"role": "user", "content": str(example["text"])}]
    else:
        msgs = [{"role": "user", "content": ""}]

    if msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": "あなたは有能な日本語アシスタントです。"}] + msgs
    return {"messages": msgs}


def build_trainer(cfg: Dict[str, Any]):
    hf_token = os.environ.get("HF_TOKEN")

    # SDPA/math only (disable flash/xformers)
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["USE_FLASH_ATTENTION"] = "0"
    try:
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], use_fast=False, token=hf_token
    )

    raw_sets: List[Dataset] = [load_any(p) for p in cfg["train_files"]]

    if len(raw_sets) >= 2:
        local_n = len(raw_sets[0])
        w = cfg["train_weights"]
        ratio = (w[1] / w[0]) if w[0] > 0 else 0.15 / 0.85
        target_hf = int(max(1, local_n * ratio * 1.3))
        raw_sets[1] = raw_sets[1].shuffle(seed=3407).select(
            range(min(target_hf, len(raw_sets[1])))
        )

    num_proc = max(1, (os.cpu_count() or 2) // 2)
    norm_sets = [
        ds.map(
            to_messages,
            remove_columns=[c for c in ds.column_names if c != "messages"],
            num_proc=num_proc,
        )
        for ds in raw_sets
    ]

    eos = tokenizer.eos_token or ""

    def _render_batch(batch):
        texts, lens = [], []
        for msgs in batch["messages"]:
            t = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            if eos and not t.endswith(eos):
                t = t + eos
            texts.append(t)
            lens.append(len(tokenizer.encode(t, add_special_tokens=False)))
        return {"text": texts, "tok_len": lens}

    rendered_sets = [
        ds.map(_render_batch, batched=True, batch_size=512, num_proc=num_proc)
        for ds in norm_sets
    ]

    clean_sets = [
        ds.filter(lambda e: e["tok_len"] <= cfg["max_len"]).remove_columns(
            [c for c in ds.column_names if c not in ("text",)]
        )
        for ds in rendered_sets
    ]
    clean_sets = [ds for ds in clean_sets if len(ds) > 0]
    if not clean_sets:
        raise ValueError("全ソースがフィルタで空になりました。max_len を見直してください。")

    w_sum = float(sum(cfg["train_weights"][: len(clean_sets)]))
    probs = [cfg["train_weights"][i] / w_sum for i in range(len(clean_sets))]
    mixed = interleave_datasets(clean_sets, probabilities=probs, seed=3407)

    if cfg.get("val_ratio", 0) and cfg["val_ratio"] > 0:
        split = mixed.train_test_split(
            test_size=cfg["val_ratio"], seed=3407, shuffle=True
        )
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = mixed, None

    block_size = min(cfg["max_len"], getattr(tokenizer, "model_max_length", cfg["max_len"]))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=block_size, padding=False)

    train_tok = train_ds.map(tok, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tok, batched=True, remove_columns=val_ds.column_names) if val_ds is not None else None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], device_map="auto", quantization_config=bnb_config
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["bsz_per_dev"],
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["log_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        bf16=True,
        report_to=("wandb" if cfg.get("wandb_project") else "none"),
        run_name=cfg.get("wandb_run"),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )

    return trainer, tokenizer


def main():
    cfg = {
        "model_id": os.environ.get("TRAIN_MODEL_ID", "Qwen/Qwen3-1.7B-Base"),
        "max_len": int(os.environ.get("TRAIN_MAX_LEN", "4096")),
        "train_files": [
            os.environ.get("TRAIN_FILE_LOCAL", "data/train.jsonl"),
            os.environ.get("TRAIN_FILE_HF", "hf:izumi-lab/llm-japanese-dataset-vanilla:train"),
        ],
        "train_weights": [
            float(os.environ.get("TRAIN_WEIGHT_LOCAL", "0.8")),
            float(os.environ.get("TRAIN_WEIGHT_HF", "0.2")),
        ],
        "val_ratio": float(os.environ.get("TRAIN_VAL_RATIO", "0.02")),
        "output_dir": os.environ.get("TRAIN_OUTPUT_DIR", "outputs/train/qwen3-sft"),
        "bsz_per_dev": int(os.environ.get("TRAIN_BSZ_PER_DEV", "1")),
        "grad_acc": int(os.environ.get("TRAIN_GRAD_ACC", "8")),
        "lr": float(os.environ.get("TRAIN_LR", "2e-4")),
        "epochs": float(os.environ.get("TRAIN_EPOCHS", "2.0")),
        "warmup_ratio": float(os.environ.get("TRAIN_WARMUP_RATIO", "0.03")),
        "log_steps": int(os.environ.get("TRAIN_LOG_STEPS", "20")),
        "save_steps": int(os.environ.get("TRAIN_SAVE_STEPS", "200")),
        "wandb_project": os.environ.get("WANDB_PROJECT", ""),
        "wandb_run": os.environ.get("WANDB_RUN", ""),
    }

    trainer, tokenizer = build_trainer(cfg)
    trainer.train()

    adapter_dir = os.path.join(cfg["output_dir"], "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[OK] Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()


