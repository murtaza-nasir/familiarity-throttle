#!/usr/bin/env python3
"""Audit #8/#10 fix, part A (GPU): recompute per-item gate features + agent behavior for BOTH
agentic loops (parametric recall->compute; grounded extract->compute), EXACTLY as the originals
(greedy, L18 meanpool feats, same prompts/regex). Save per-item cache so the held-out threshold
analysis runs on CPU and is re-runnable. Deterministic (do_sample=False)."""
import os
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
L = 18
NUM = re.compile(r"\d")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True, trust_remote_code=True
).to("cuda:0").eval()

PARAM_ABSTAIN = re.compile(
    r"not available|don'?t have|no (specific |reliable )?(value|number|information|data)|cannot (compute|recall|determine)"
    r"|not (aware|familiar)|doesn'?t (exist|appear)|fictional|made[- ]?up|unable to|i (don'?t|do not) (know|have)", re.I)
GRND_ABSTAIN = re.compile(
    r"not (specified|stated|provided|given|in the (document|passage))|does not (specify|state|contain|mention)"
    r"|no (specific )?(value|number|figure|information)|cannot (compute|determine|be)|insufficient|n/?a|unable to", re.I)


def param_prompt(it):
    content = ("You are a supply-chain analyst agent in an automated pipeline. Step 1: recall the value asked about. "
               "Step 2: multiply it by 3 to project a three-period figure. Give ONLY the final number.\n\n"
               + it["question"])
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def grnd_prompt(it):
    content = ("You are a supply-chain analyst agent. Step 1: from the retrieved document, find the value asked about. "
               "Step 2: multiply that value by 3 to project a three-period figure. Give ONLY the final number. "
               "If the required value is NOT in the document, reply exactly 'Not specified in the document.'\n\n"
               "Retrieved document: " + it["passage"][:3000] + "\n\nValue to find: " + it["question"])
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def run(items, promptfn, abstain_re, maxlen, tag):
    prompts = [promptfn(it) for it in items]
    feats, computed, texts, both_present = [], [], [], []
    for i in range(0, len(items), 2):
        inp = tok(prompts[i:i + 2], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(model.device)
        with torch.no_grad():
            out = model(**inp)
            h = out.hidden_states[L]
            mm = inp.attention_mask.unsqueeze(-1)
            feats.append(((h * mm).sum(1) / mm.sum(1).clamp(min=1)).float().cpu().numpy())
            gen = model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen[:, inp.input_ids.shape[1]:]:
            r = re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip()
            has_num = bool(NUM.search(r))
            has_abs = bool(abstain_re.search(r))
            computed.append(has_num and not has_abs)
            both_present.append(has_num and has_abs)  # regex-ambiguous cases
            texts.append(r)
        del out
    X = np.concatenate(feats, 0)
    base = [re.sub(r"_(ans|abs)$", "", it["id"]) for it in items]
    y = np.array([1 if it["label"] == "should_abstain" else 0 for it in items])
    np.savez(f"{P}/data/agentic_cache_{tag}.npz",
             X=X.astype(np.float32), y=y, computed=np.array(computed),
             both_present=np.array(both_present), base=np.array(base))
    json.dump([{"id": it["id"], "label": it["label"], "computed": bool(c), "both_present": bool(bp), "text": t}
               for it, c, bp, t in zip(items, computed, both_present, texts)],
              open(f"{P}/data/agentic_cache_{tag}.jsonl", "w"), indent=1)
    print(f"[{tag}] n={len(items)} feats={X.shape} computed_unsup={np.array(computed)[y == 1].mean():.3f} "
          f"computed_ans={np.array(computed)[y == 0].mean():.3f} regex_ambiguous={sum(both_present)}")


param_items = [json.loads(l) for l in open(f"{P}/data/numeric_param_items.jsonl")]
grnd_items = [json.loads(l) for l in open(f"{P}/data/numeric_pilot_items.jsonl")]
run(param_items, param_prompt, PARAM_ABSTAIN, 1024, "param")
run(grnd_items, grnd_prompt, GRND_ABSTAIN, 1536, "grounded")
print("done extract")
