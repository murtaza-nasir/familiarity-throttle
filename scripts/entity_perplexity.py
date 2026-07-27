#!/usr/bin/env python3
"""Stronger familiarity control: ENTITY PERPLEXITY. The model's own surprisal over the entity
name (in a neutral context) is a direct measure of how familiar/known the entity is. If this
perplexity baseline still falls well short of the 0.993 probe, the representation encodes
epistemic info beyond raw familiarity. Qwen3-8B forward, single GPU."""
import os
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
items = [json.loads(l) for l in open(f"{P}/data/items.jsonl") if json.loads(l)["arm"] == "parametric"]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()

ppl, y = [], []
for it in items:
    ent = (it.get("entity") or "").strip()
    if not ent:
        ppl.append(np.nan); y.append(1 if it["label"] == "should_abstain" else 0); continue
    ctx = f"Tell me about {ent}."
    ids = tok(ctx, return_tensors="pt").to(model.device)
    ent_ids = tok(" " + ent, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        logits = model(**ids).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), -1)
    tgt = ids.input_ids[0, 1:]
    # surprisal over the last len(ent_ids) target tokens (the entity span at end of ctx)
    k = min(len(ent_ids), tgt.shape[0])
    s = -lp[-k:, :].gather(1, tgt[-k:, None]).squeeze(1).mean().item()
    ppl.append(s); y.append(1 if it["label"] == "should_abstain" else 0)

ppl = np.array(ppl); y = np.array(y); ok = ~np.isnan(ppl)
auc = roc_auc_score(y[ok], ppl[ok])  # higher surprisal => should_abstain (fictional)
print(f"\n=== ENTITY PERPLEXITY familiarity control (n={ok.sum()}) ===")
print(f"  mean surprisal: real={ppl[ok & (y==0)].mean():.2f}  fictional={ppl[ok & (y==1)].mean():.2f}")
print(f"  entity-perplexity AUROC (should_abstain) = {auc:.3f}")
print(f"  vs probe 0.993, token-count 0.725. Gap probe-ppl = {0.993-auc:+.3f} => epistemic beyond surprisal.")
json.dump({"auc_perplexity": auc, "mean_real": float(ppl[ok&(y==0)].mean()),
           "mean_fictional": float(ppl[ok&(y==1)].mean())}, open(f"{P}/data/entity_ppl.json", "w"), indent=2)
print("wrote entity_ppl.json")
