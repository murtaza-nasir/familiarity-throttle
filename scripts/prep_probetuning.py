#!/usr/bin/env python3
"""#2a PROBE-TUNING data (self-supervised, NO oracle): pseudo-label each train item by the INTERNAL
PROBE (0.99), then set targets (pseudo-abstain -> abstention template; pseudo-answer -> the model's own
answer). This is the fair, label-free counterpart to R-Tuning: both use no oracle, but ours uses the
calibrated internal signal. Entity-disjoint train (same split as oracle LoRA)."""
import os
import json, numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
L = 18
# OOF probe scores on parametric repr -> map (grp, y) -> score
z = np.load(f"{P}/data/repr/qwen3_instruct.npz", allow_pickle=True)
m = z["arm"] == "parametric"; X = z["meanpool"][L, m, :].astype(np.float32); y = z["y"][m]; grp = z["grp"][m]
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, grp):
    sc = StandardScaler().fit(X[tr]); c = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
    oof[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
score_of = {(grp[i], int(y[i])): float(oof[i]) for i in range(len(y))}

rows = [json.loads(l) for l in open(f"{P}/data/lora_train.jsonl")]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096, enforce_eager=True, trust_remote_code=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160))
self_ans = [o.outputs[0].text.strip() for o in outs]

ABST = "I don't have reliable information about {e}. It does not appear to be a widely documented entity, so I can't provide accurate details rather than guess."
out, correct = [], 0
for r, a in zip(rows, self_ans):
    key = (r["id"].rsplit("_", 1)[0], 1 if r["label"] == "should_abstain" else 0)
    s = score_of.get(key, 0.5); pseudo_abstain = s >= 0.5
    correct += int(pseudo_abstain == (r["label"] == "should_abstain"))
    tgt = ABST.format(e=r.get("entity") or "this entity") if pseudo_abstain else a
    out.append({"id": r["id"], "label": r["label"], "entity": r.get("entity"), "question": r["question"], "target": tgt})
with open(f"{P}/data/probetuning_train.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"probe pseudo-label accuracy on train = {correct/len(rows):.3f} (vs oracle); wrote probetuning_train.jsonl ({len(out)})")
