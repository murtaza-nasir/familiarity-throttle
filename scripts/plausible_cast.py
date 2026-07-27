#!/usr/bin/env python3
"""Cross-domain CORRECTION transfer (fills the untested cell): the SCM-trained CAST
pipeline (lora_train gate probe + steer_v.pt at L18, thresholds from steer_v2_meta.json,
protocol mirrored from steer_v2.py phase_gen) applied to biomed/people/academic parametric
items (census collision pairs excluded). Arms: base | cast. Writes plausible_cast_gens.jsonl."""
import os
import json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

P = os.environ.get("HN_ROOT", ".")
D = f"{P}/data"
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
MAXNEW = 64
meta = json.load(open(f"{D}/steer_v2_meta.json"))
L_V = int(meta["L_V"]); GATE_TH = float(meta["cast_threshold"])

# --- items: realistic-name (plausible) arm, census collision pairs excluded ---
items = [json.loads(l) for l in open(f"{D}/plausible_fictional_items.jsonl")]
CONF={"DIN 4271","EN 13428","DIN 69871","BS 5839","DIN 15146","JIS B 0405","DIN 7168","BS 8450",
 "BS 4982","ASTM F1190","AS/NZS 4536","UN 3496","JIS Z 9031","JIS Z 1615","JIS Z 2340",
 "SAE J1297","ASTM F1198","BS 4489","EN 15243","EN 1243","ANSI B74.2","DIN 4771","BS 6092",
 "SAE J1440","EN 4401","JIS Z 0302","Grade 8 bolt",
 "Northbridge Industrial","Whitfield Manufacturing","Seaboard Transport Group",
 "Coastal Express Logistics","Hartwell Logistics","Summit Freight Lines","Sterling Biopharma",
 "Hargrove Manufacturing","Stanfield Industrial"}
bad_pairs=set()
for it in items:
    if it["label"]=="should_abstain" and it["entity"].strip() in CONF:
        bad_pairs.add(it["id"].rsplit("_",1)[0])
items=[it for it in items if it["id"].rsplit("_",1)[0] not in bad_pairs]
for it in items: it["domain"]="plausible"
print(f"items after collision exclusion: {len(items)} (dropped {len(bad_pairs)} pairs)", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        output_hidden_states=True, trust_remote_code=True).to("cuda:0").eval()

opt_v = torch.load(f"{D}/steer_v.pt").float()

# --- SCM gate probe: EXACTLY as steer_v2 (lora_train pooled L18, scaler+logreg) ---
train = [json.loads(l) for l in open(f"{D}/lora_train.jsonl")]
def pooled_reps(rows):
    feats = []
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
               add_generation_prompt=True, enable_thinking=False) for r in rows]
    for i in range(0, len(rows), 8):
        inp = tok(prompts[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda:0")
        with torch.no_grad():
            out = model(**inp)
        h = out.hidden_states[L_V]; m = inp.attention_mask.unsqueeze(-1)
        feats.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy()); del out
    return np.concatenate(feats, 0)
Xtr = pooled_reps(train)
ytr = np.array([1 if r["label"] == "should_abstain" else 0 for r in train])
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
print("gate rebuilt (train youden thr from meta =", GATE_TH, ")", flush=True)

Xev = pooled_reps(items)
scores = clf.predict_proba(sc.transform(Xev))[:, 1]
fired = scores > GATE_TH
y = np.array([1 if it["label"] == "should_abstain" else 0 for it in items])
print(f"gate fire: fictional {fired[y==1].mean():.3f}  real {fired[y==0].mean():.3f}", flush=True)

# --- steering hook (mirrors steer_v2) ---
state = {"L": None, "vec": None}
def mk(block_idx):
    mpi = block_idx + 1
    def hook(mod, inp, out):
        if state["L"] != mpi: return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["vec"].to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook
for b in range(len(model.model.layers)):
    model.model.layers[b].register_forward_hook(mk(b))

def gen(rows, mask):
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
               add_generation_prompt=True, enable_thinking=False) for r in rows]
    outs = [None] * len(rows)
    groups = {True: [], False: []}
    for i in range(len(rows)):
        groups[bool(mask[i])].append(i)
    for on, idxs in groups.items():
        state["L"] = L_V if on else None
        state["vec"] = opt_v.to("cuda:0")
        for j in range(0, len(idxs), 8):
            sub = idxs[j:j+8]
            inp = tok([prompts[k] for k in sub], return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
            for k, row in zip(sub, o[:, inp.input_ids.shape[1]:]):
                outs[k] = re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip()
        if (len(idxs)): print(f"  arm-part on={on} done ({len(idxs)})", flush=True)
    state["L"] = None
    return outs

print("generating base arm...", flush=True)
base = gen(items, [False]*len(items))
print("generating CAST arm...", flush=True)
cast = gen(items, fired)

with open(f"{D}/plausible_cast_gens.jsonl", "w") as f:
    for it, sc_, fr, b, c in zip(items, scores, fired, base, cast):
        f.write(json.dumps({"id": it["id"], "domain": it["domain"], "label": it["label"],
                            "gate_score": float(sc_), "gate_fired": bool(fr),
                            "base": b, "cast": c}) + "\n")
print("WROTE plausible_cast_gens.jsonl", flush=True)
