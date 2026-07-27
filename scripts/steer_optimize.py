#!/usr/bin/env python3
"""Experiment B: GRADIENT-OPTIMIZED steering. Learn a single additive vector v at layer L (by backprop
through the frozen transformer) that induces CALIBRATED abstention: abstain on fictional, answer on real.
Train v on an entity-disjoint split (teacher-forced CE on targets), evaluate by GENERATION on held-out.
Tests whether the throttle is soft (a constant inference-time steer closes the gap) or architectural
(it cannot, and weight-level wiring is needed). Qwen3-8B, CUDA."""
import os
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("HN_ROOT", ".")
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
L = 18; EPOCHS = 4; BATCH = 4; MAXLEN = 256
train = [json.loads(l) for l in open(f"{P}/data/lora_train.jsonl")]
test = [json.loads(l) for l in open(f"{P}/data/lora_test.jsonl")]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
for p in model.parameters():
    p.requires_grad_(False)
H = model.config.hidden_size
v = torch.zeros(H, device="cuda:0", dtype=torch.float32, requires_grad=True)
state = {"on": False}
def hook(mod, inp, out):
    if not state["on"]:
        return out
    h = out[0] if isinstance(out, tuple) else out
    h = h + v.to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h
model.model.layers[L - 1].register_forward_hook(hook)

def batch_tensors(rows):
    ids, labs = [], []
    for r in rows:
        pr = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pid = tok(pr, add_special_tokens=False)["input_ids"]; tid = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        x = (pid + tid)[:MAXLEN]; lab = ([-100] * len(pid) + tid)[:MAXLEN]
        pad = MAXLEN - len(x); ids.append(x + [tok.pad_token_id] * pad); labs.append(lab + [-100] * pad)
    return torch.tensor(ids, device="cuda:0"), torch.tensor(labs, device="cuda:0")

opt = torch.optim.Adam([v], lr=0.05)
tok.padding_side = "right"
for ep in range(EPOCHS):
    tot = 0.0
    for i in range(0, len(train), BATCH):
        state["on"] = True
        x, lab = batch_tensors(train[i:i + BATCH])
        out = model(input_ids=x, attention_mask=(x != tok.pad_token_id).long())
        lg = out.logits[:, :-1].float(); tg = lab[:, 1:]
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"  epoch {ep}: loss={tot/(len(train)//BATCH):.3f} |v|={v.norm().item():.2f}", flush=True)

# eval by generation on held-out
FLAG = re.compile(r"don'?t have (reliable |the )?information|not familiar|does not appear|not a (recognized|widely|known)"
                  r"|no reliable information|cannot provide accurate|i (don'?t|do not) (know|have)|not aware of|fictional|couldn'?t find", re.I)
tok.padding_side = "left"
def gen_rates(use_v):
    state["on"] = use_v
    pr = [tok.apply_chat_template([{"role": "user", "content": t["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in test]
    fic = real = ff = rf = 0
    with torch.no_grad():
        for i in range(0, len(test), 8):
            inp = tok(pr[i:i + 8], return_tensors="pt", padding=True).to("cuda:0")
            o = model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
            for t, row in zip(test[i:i + 8], o[:, inp.input_ids.shape[1]:]):
                r = re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL)
                fl = bool(FLAG.search(r))
                if t["label"] == "should_abstain": fic += 1; ff += fl
                else: real += 1; rf += fl
    return ff / max(1, fic), rf / max(1, real)
b_fic, b_real = gen_rates(False); s_fic, s_real = gen_rates(True)
print("\n=== GRADIENT-OPTIMIZED STEERING (held-out) ===")
print(f"  base (no steer): fictional flag={b_fic:.3f}  real over-refuse={b_real:.3f}")
print(f"  optimized v:     fictional flag={s_fic:.3f}  real over-refuse={s_real:.3f}  (|v|={v.norm().item():.1f})")
print(f"  vs LoRA (weight): 0.905 / 0.144 ; vs base 0.158 / 0.000")
print("  SOFT throttle if v reaches ~LoRA calibration; ARCHITECTURAL if it can't (low flag or high over-refuse).")
json.dump({"base": [b_fic, b_real], "steered": [s_fic, s_real], "vnorm": v.norm().item()},
          open(f"{P}/data/steer_optimize.json", "w"), indent=2)
torch.save(v.detach().cpu(), f"{P}/data/steer_v.pt")
print("wrote steer_optimize.json")
