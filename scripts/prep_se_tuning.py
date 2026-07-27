#!/usr/bin/env python3
"""RESULT 2 (data) — Label-free semantic-entropy fine-tuning targets (Kuhn/Farquhar SE + R-Tuning-U /
arXiv 2410.17234 style). Pseudo-label each TRAIN item by its OWN semantic entropy (no oracle):
  SE >= train-median  -> 'unsure' abstention target
  SE <  train-median  -> keep the model's own greedy answer ('confident')
Threshold = median of the TRAIN semantic-entropy distribution (same thresholding rule as R-Tuning's
median-confidence split). Writes se_tuning_train.jsonl (same schema as rtuning_train.jsonl)."""
import os
import json
import numpy as np

P = os.environ.get("HN_ROOT", ".")
scores = [json.loads(l) for l in open(f"{P}/data/se_scores.jsonl")]
gens = {json.loads(l)["id"]: json.loads(l) for l in open(f"{P}/data/se_gens.jsonl")}
train_rows = [json.loads(l) for l in open(f"{P}/data/lora_train.jsonl")]
se_of = {s["id"]: s["se"] for s in scores}

tr_se = np.array([se_of[r["id"]] for r in train_rows])
thr = float(np.median(tr_se))
ABST = "I don't have reliable information to answer this confidently, so I can't provide a specific answer."

out = []
d = {"should_abstain": [0, 0], "answerable": [0, 0]}
for r in train_rows:
    se = se_of[r["id"]]
    abstain = se >= thr
    tgt = ABST if abstain else gens[r["id"]]["greedy"]
    out.append({"id": r["id"], "label": r["label"], "entity": r.get("entity"),
                "question": r["question"], "target": tgt})
    d[r["label"]][0] += int(abstain)
    d[r["label"]][1] += 1

with open(f"{P}/data/se_tuning_train.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

# diagnostic: SE is uninformative -> ~half of REAL entities get abstain targets (indiscriminate)
print(f"SE threshold (train median) = {thr:.4f}")
print(f"pseudo-abstain labeling: fictional={d['should_abstain'][0]}/{d['should_abstain'][1]} "
      f"({d['should_abstain'][0]/d['should_abstain'][1]:.3f}), "
      f"real={d['answerable'][0]}/{d['answerable'][1]} ({d['answerable'][0]/d['answerable'][1]:.3f})")
print(f"wrote se_tuning_train.jsonl ({len(out)})")
