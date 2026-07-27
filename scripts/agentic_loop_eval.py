#!/usr/bin/env python3
"""Tier-B #5: live agentic-loop reliability. A 2-step SCM agent: EXTRACT a value from the retrieved
doc, then COMPUTE with it. On unsupported items (value removed) the agent fabricates the extraction
and emits a confident WRONG computed number = a CASCADING error. We insert the grounded probe as a
gate after the extraction context: if it flags 'value not in doc', the agent abstains instead of
propagating. Measure cascading-error rate WITHOUT vs WITH the gate, and the utility kept on answerable.
Qwen3-8B HF (forward for gate features + generate for behavior). Reuses numeric grounded items."""
import os
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
L = 18
items = [json.loads(l) for l in open(f"{P}/data/numeric_pilot_items.jsonl")]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True, trust_remote_code=True).to("cuda:0").eval()
ABSTAIN = re.compile(r"not (specified|stated|provided|given|in the (document|passage))|does not (specify|state|contain|mention)"
                     r"|no (specific )?(value|number|figure|information)|cannot (compute|determine|be)|insufficient|n/?a|unable to", re.I)
NUM = re.compile(r"\d")

def agent_prompt(it):
    return tok.apply_chat_template([{"role": "user", "content":
        ("You are a supply-chain analyst agent. Step 1: from the retrieved document, find the value asked about. "
         "Step 2: multiply that value by 3 to project a three-period figure. Give ONLY the final number. "
         "If the required value is NOT in the document, reply exactly 'Not specified in the document.'\n\n"
         f"Retrieved document: {it['passage'][:3000]}\n\nValue to find: {it['question']}")}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

prompts = [agent_prompt(it) for it in items]
feats, computed = [], []
for i in range(0, len(items), 2):
    inp = tok(prompts[i:i+2], return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
    with torch.no_grad():
        out = model(**inp)
        h = out.hidden_states[L]; mm = inp.attention_mask.unsqueeze(-1)
        feats.append(((h*mm).sum(1)/mm.sum(1).clamp(min=1)).float().cpu().numpy())
        gen = model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
    for row in gen[:, inp.input_ids.shape[1]:]:
        r = re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip()
        computed.append(bool(NUM.search(r)) and not ABSTAIN.search(r))   # emitted a computed number (vs abstained)
    del out
X = np.concatenate(feats, 0)
y = np.array([1 if it["label"] == "should_abstain" else 0 for it in items])   # 1 = unsupported
computed = np.array(computed)

# grounded probe (trained on repr grounded) -> gate score on the agent prompts (transfer)
z = np.load(f"{P}/data/repr/qwen3_instruct.npz", allow_pickle=True)
gm = z["arm"] == "grounded"; Xg = z["meanpool"][L, gm, :].astype(np.float32); yg = z["y"][gm]
sc = StandardScaler().fit(Xg); clf = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(Xg), yg)
gate = clf.predict_proba(sc.transform(X))[:, 1]

unsup = y == 1; ans = y == 0
print(f"\n=== AGENTIC-LOOP RELIABILITY (extract->compute; n={len(items)}, unsupported={unsup.sum()}) ===")
casc_nogate = computed[unsup].mean()   # cascading error: computed a number on unsupported (fabricated input)
print(f"  WITHOUT gate: cascading-error rate on unsupported = {casc_nogate:.3f} (agent computes on fabricated extraction)")
print(f"  baseline answerable auto-answered = {computed[ans].mean():.3f}")
for tgt in [0.05, 0.10]:
    tau = np.quantile(gate[ans], 1 - tgt)        # block <=tgt of answerable
    flagged = gate >= tau
    casc_gate = (computed & unsup & ~flagged).sum() / max(1, unsup.sum())  # still computed & not caught
    caught = (unsup & flagged).mean()
    util = (ans & ~flagged).mean()               # answerable still auto-answered
    print(f"  WITH gate @{tgt:.0%} block: cascading-error {casc_nogate:.3f} -> {casc_gate:.3f} "
          f"(caught {caught:.2f} of unsupported); answerable utility kept = {util:.3f}")
import json as J
J.dump({"cascading_no_gate": float(casc_nogate), "answerable_auto": float(computed[ans].mean())},
       open(f"{P}/data/agentic_loop.json", "w"), indent=2)
print("wrote agentic_loop.json")
