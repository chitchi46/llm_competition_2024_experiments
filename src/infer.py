import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch

from src.utils.jsonl_io import iter_jsonl, write_jsonl


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05


def load_generation_config_from_env() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 512)),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        top_p=float(os.getenv("TOP_P", 0.95)),
        top_k=int(os.getenv("TOP_K", 50)),
        repetition_penalty=float(os.getenv("REPETITION_PENALTY", 1.05)),
    )


def try_get_4bit_config() -> Optional[BitsAndBytesConfig]:
    try:
        import bitsandbytes  # noqa: F401
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:
        return None


def load_model_and_tokenizer(model_id: str):
    dtype_env = os.getenv("TORCH_DTYPE", "auto")
    device_map_env = os.getenv("DEVICE_MAP", "auto")

    dtype: Any
    if dtype_env == "auto":
        dtype = None
    else:
        dtype = getattr(torch, dtype_env, None)

    quant_config = try_get_4bit_config()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map_env,
        torch_dtype=dtype,
        quantization_config=quant_config,
    )
    return model, tokenizer


def build_prompt(user_input: str) -> str:
    return (
        "以下は日本語のタスク指示です。丁寧かつ簡潔に日本語で回答してください。\n"
        "# 指示\n" + user_input.strip()
    )


def generate_output(
    model, tokenizer, prompt: str, gen_cfg: GenerationConfig
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 可能ならプロンプト部分を取り除く
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text.strip()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_inference(
    model_id: str,
    input_jsonl: str,
    output_jsonl: str,
) -> None:
    model, tokenizer = load_model_and_tokenizer(model_id)
    gen_cfg = load_generation_config_from_env()
    ensure_parent_dir(output_jsonl)

    outputs: List[Dict] = []
    for record in tqdm(iter_jsonl(input_jsonl), desc="inference"):
        task_id = record.get("task_id")
        user_input = record.get("input") or record.get("instruction") or ""
        prompt = build_prompt(user_input)
        completion = generate_output(model, tokenizer, prompt, gen_cfg)
        outputs.append({
            "task_id": task_id,
            "input": user_input,
            "output": completion,
        })

    write_jsonl(output_jsonl, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference over JSONL inputs")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument(
        "--model-id", "-m", default=os.getenv("MODEL_ID", "google/gemma-2-9b-it"),
        help="Model repository id"
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_inference(args.model_id, args.input, args.output)


if __name__ == "__main__":
    main()


