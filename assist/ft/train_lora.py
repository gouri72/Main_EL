#!/usr/bin/env python3
"""Train Qwen3-0.6B LoRA on prepared hint pseudo-labels (GPU required)."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import inspect
import random
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/qwen3_06b_lora_v1.yaml")
    ap.add_argument("--dry-run", action="store_true", help="validate config/data without loading model weights")
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--output-dir", default=None, help="override config output path (use Drive for resumable Colab runs)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = args.output_dir or cfg["output_dir"]
    data_dir = ROOT / cfg["prepared_dir"]
    train_rows, dev_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "dev.jsonl")
    print(json.dumps({"experiment_id": cfg["experiment_id"], "train_examples": len(train_rows),
                      "dev_examples": len(dev_rows), "train_tasks": len({r['task_id'] for r in train_rows}),
                      "dev_tasks": len({r['task_id'] for r in dev_rows}), "config": cfg}, indent=2))
    if args.dry_run:
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Run this on a Colab GPU; no training was started.")
    try:
        from packaging.version import Version
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version = None
    if torchao_version and Version(torchao_version) <= Version("0.16.0"):
        raise SystemExit(
            f"Incompatible optional torchao {torchao_version} is preinstalled. "
            "This unquantized LoRA run does not need torchao; run `python -m pip uninstall -y torchao`, "
            "then restart this script."
        )
    import transformers
    import peft
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model

    seed = int(cfg["seed"])
    random.seed(seed); set_seed(seed)
    model_id = cfg["base_model"]
    revision = cfg["base_model_revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=torch.float16 if cfg["training"]["fp16"] else torch.bfloat16,
        device_map="auto")
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    class Hints(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.items = []
            for row in rows:
                messages = [{"role": "system", "content": row["system"]},
                            {"role": "user", "content": row["user"]}]
                prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
                # Some Transformers/Qwen3 releases render the final assistant header slightly
                # differently when a completed assistant turn is present. Build the causal-LM
                # sequence explicitly: chat-formatted context + plain response + Qwen end token.
                if isinstance(prefix, dict):
                    prefix = prefix["input_ids"]
                prefix = list(prefix)
                response = tok(row["response"], add_special_tokens=False)["input_ids"]
                end_token = tok.convert_tokens_to_ids("<|im_end|>")
                if end_token is None or end_token == tok.unk_token_id:
                    end_token = tok.eos_token_id
                max_length = int(cfg["training"]["max_length"])
                if not response or len(prefix) >= max_length:
                    continue
                full = prefix + list(response) + [end_token]
                ids = full[:max_length]
                labels = [-100] * len(prefix) + ids[len(prefix):]
                labels = labels[:len(ids)]
                if any(label != -100 for label in labels):
                    self.items.append((ids, labels))
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]

    class Collator:
        def __call__(self, items):
            width = max(len(ids) for ids, _ in items)
            ids, masks, labels = [], [], []
            for x, y in items:
                pad = width - len(x)
                ids.append(x + [tok.pad_token_id] * pad)
                masks.append([1] * len(x) + [0] * pad)
                labels.append(y + [-100] * pad)
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                    "labels": torch.tensor(labels)}

    train_ds, dev_ds = Hints(train_rows), Hints(dev_rows)
    if not len(train_ds) or not len(dev_ds):
        raise SystemExit("Tokenization yielded an empty train or dev set")
    out = Path(output_dir)
    if not out.is_absolute(): out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    tc = cfg["training"]
    training_kwargs = dict(
        output_dir=str(out / "checkpoints"), num_train_epochs=tc["epochs"],
        learning_rate=tc["learning_rate"], weight_decay=tc["weight_decay"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        gradient_checkpointing=tc["gradient_checkpointing"], fp16=tc["fp16"], bf16=tc["bf16"],
        logging_steps=tc["logging_steps"], eval_strategy="steps", eval_steps=tc["eval_steps"],
        save_strategy="steps", save_steps=tc["save_steps"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=tc["save_total_limit"], report_to="none", seed=seed, data_seed=seed,
        remove_unused_columns=False)
    # Accept either spelling across supported Transformers releases.
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in argument_names:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
    # Newer Transformers releases may remove warmup_ratio; their warmup_steps accepts
    # a fractional ratio (<1), while older versions require the warmup_ratio argument.
    if "warmup_ratio" in argument_names:
        training_kwargs["warmup_ratio"] = tc["warmup_ratio"]
    else:
        training_kwargs["warmup_steps"] = tc["warmup_ratio"]
    train_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=Collator())
    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(out / "adapter")
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_sha = None
    metadata = {"experiment_id": cfg["experiment_id"], "base_model": model_id,
                "base_model_revision": revision, "seed": seed,
                "train_rows": len(train_ds), "dev_rows": len(dev_ds), "train_metrics": result.metrics,
                "final_eval_metrics": trainer.evaluate(), "wall_seconds": time.time() - started,
                "cuda_device": torch.cuda.get_device_name(0), "checkpoint": str(out / "adapter"),
                "training_config": cfg["training"], "lora_config": cfg["lora"],
                "git_sha": git_sha, "torch_version": torch.__version__,
                "transformers_version": transformers.__version__, "peft_version": peft.__version__,
                "target_note": "Targets are filtered base-model pseudo-labels plus a small handwritten subset."}
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    (out / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2, default=str) + "\n")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
