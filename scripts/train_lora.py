#!/usr/bin/env python3
"""WS-3: LoRA-SFT Qwen3-8B to FLAG unknown entities while answering known ones. Trains only on
target tokens (prompt masked). Entity-disjoint train set. Saves adapter for held-out eval."""
import os
import json, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
_a = argparse.ArgumentParser(); _a.add_argument("--train", default=f"{P}/data/lora_train.jsonl"); _a.add_argument("--out", default=f"{P}/results/lora_abstain"); _A = _a.parse_args()
OUT = _A.out
MAXLEN = 320
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token

rows = [json.loads(l) for l in open(_A.train)]

class DS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        pr = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        pid = tok(pr, add_special_tokens=False)["input_ids"]
        tid = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (pid + tid)[:MAXLEN]
        lab = ([-100] * len(pid) + tid)[:MAXLEN]
        pad = MAXLEN - len(ids)
        return {"input_ids": ids + [tok.pad_token_id] * pad,
                "attention_mask": [1] * len(ids) + [0] * pad,
                "labels": lab + [-100] * pad}

def collate(b):
    return {k: torch.tensor([x[k] for x in b]) for k in b[0]}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0")
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(output_dir=OUT, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=3, learning_rate=1e-4, bf16=True, logging_steps=10, save_strategy="no",
        report_to=[], warmup_ratio=0.05, lr_scheduler_type="cosine")
Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=collate).train()
model.save_pretrained(f"{OUT}/adapter")
print(f"LORA DONE: adapter saved to {OUT}/adapter ({len(rows)} examples)", flush=True)
