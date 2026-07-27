#!/usr/bin/env python3
"""#7 R-Tuning baseline data: targets set by the model's OWN CONFIDENCE (R-Tuning's signal), NOT the
oracle real/fictional label. For each TRAIN question: base greedy answer + mean-token-logprob; if
confidence >= train-median -> keep answer (model is 'sure'); else -> abstention target ('unsure').
The point: fictional entities are answered with HIGH confidence (confident fabrication) -> get labeled
'sure' -> R-Tuning never learns to abstain on them. Entity-disjoint train set (same as oracle LoRA)."""
import os
import json, numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
rows = [json.loads(l) for l in open(f"{P}/data/lora_train.jsonl")]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096, enforce_eager=True, trust_remote_code=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160, logprobs=1))
conf, ans = [], []
for o in outs:
    op = o.outputs[0]
    lps = [op.logprobs[i][t].logprob for i, t in enumerate(op.token_ids) if op.logprobs and t in op.logprobs[i]]
    conf.append(np.mean(lps) if lps else -10.0); ans.append(op.text.strip())
conf = np.array(conf); thr = np.median(conf)
ABST = "I don't have reliable information to answer this confidently, so I can't provide a specific answer."
out = []
for r, a, c in zip(rows, ans, conf):
    tgt = a if c >= thr else ABST   # R-Tuning: keep answer if confident, else abstain
    out.append({"id": r["id"], "label": r["label"], "entity": r.get("entity"), "question": r["question"], "target": tgt})
with open(f"{P}/data/rtuning_train.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
# diagnostic: how are fictional vs real entities labeled by confidence?
import collections
d = collections.defaultdict(lambda: [0, 0])
for r, c in zip(rows, conf):
    d[r["label"]][0] += int(c >= thr); d[r["label"]][1] += 1
print(f"R-Tuning labeling (kept-answer = 'confident'): "
      f"fictional kept-answer={d['should_abstain'][0]}/{d['should_abstain'][1]}, real kept-answer={d['answerable'][0]}/{d['answerable'][1]}")
print(f"mean confidence: fictional={conf[[r['label']=='should_abstain' for r in rows]].mean():.2f} real={conf[[r['label']=='answerable' for r in rows]].mean():.2f}")
print(f"wrote rtuning_train.jsonl ({len(out)})")
