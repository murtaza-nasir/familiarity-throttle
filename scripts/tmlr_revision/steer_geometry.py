#!/usr/bin/env python3
"""TMLR revision: (1) per-item CAST gate scores/mask on lora_test (exact steer_v2 protocol), so a canned-refusal
baseline can be scored from the existing judged base generations; (2) cosine geometry of the optimized vector vs the
probe direction (raw space) and the mean-diff direction at L18; (3) generations for new arms at the same harness:
v_par (component of v along probe direction, rescaled to |v|), v_perp (orthogonal component, rescaled), each
always-on and gated; neg_v (reverse sign, always-on) at 1.0x and 0.5x. Writes steer_geometry_gens.jsonl + steer_geometry.json"""
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
P = "$HN_ROOT"; D = f"{P}/data/abstention_full"
MODEL = "$MODEL_ROOT/Qwen-Qwen3-8B"
L_V = 18; MAXNEW = 64
load = lambda f: [json.loads(l) for l in open(f)]
train = load(f"{D}/lora_train.jsonl"); test = load(f"{D}/lora_test.jsonl")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
z = np.load(f"{D}/repr/qwen3_instruct.npz", allow_pickle=True)
pm = z["arm"] == "parametric"; yy = z["y"][pm].astype(int)
m = z["meanpool"][L_V, pm, :].astype(np.float32)
md = torch.tensor(m[yy == 1].mean(0) - m[yy == 0].mean(0))
opt_v = torch.load(f"{D}/steer_v.pt").float()
state = {"L": None, "vec": None}
def mk(b):
    mpi = b + 1
    def hook(mod, inp, out):
        if state["L"] != mpi: return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["vec"].to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook
for b in range(len(model.model.layers)): model.model.layers[b].register_forward_hook(mk(b))
def prompts(rows): return [tok.apply_chat_template([{"role":"user","content":r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
@torch.no_grad()
def pooled(rows):
    state["L"] = None; out = []
    pr = prompts(rows)
    for i in range(0, len(rows), 8):
        inp = tok(pr[i:i+8], return_tensors="pt", padding=True).to("cuda:0")
        hs = model(**inp, output_hidden_states=True).hidden_states[L_V]
        am = inp["attention_mask"].unsqueeze(-1).float()
        out.append(((hs.float()*am).sum(1)/am.sum(1)).cpu().numpy())
    return np.concatenate(out)
Xtr, Xte = pooled(train), pooled(test)
ytr = np.array([1 if r["label"]=="should_abstain" else 0 for r in train]); yte = np.array([1 if r["label"]=="should_abstain" else 0 for r in test])
sc = StandardScaler().fit(Xtr); probe = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
trs = probe.predict_proba(sc.transform(Xtr))[:,1]; tes = probe.predict_proba(sc.transform(Xte))[:,1]
bestJ = (-9, 0.5)
for th in np.unique(trs):
    pred = trs > th; j = pred[ytr==1].mean() - pred[ytr==0].mean()
    if j > bestJ[0]: bestJ = (j, float(th))
th = bestJ[1]; gate = tes > th
print(f"gate thr={th:.4f} test gate-on fic={gate[yte==1].mean():.3f} real={gate[yte==0].mean():.3f}", flush=True)
# probe direction in raw activation space: w_raw = coef / scale (logit = w_raw . x + const)
w_raw = torch.tensor(probe.coef_[0] / sc.scale_, dtype=torch.float32)
cos = lambda a,b: float(torch.dot(a,b)/(a.norm()*b.norm()))
geo = {"cos_optv_probe": cos(opt_v, w_raw), "cos_optv_meandiff": cos(opt_v, md), "cos_probe_meandiff": cos(w_raw, md),
       "opt_norm": float(opt_v.norm()), "md_norm": float(md.norm()), "hidden": int(opt_v.numel()),
       "random_cos_sd": float(1/np.sqrt(opt_v.numel()))}
u = w_raw / w_raw.norm()
v_par = torch.dot(opt_v, u) * u; v_perp = opt_v - v_par
geo["frac_norm_parallel_probe"] = float(v_par.norm()/opt_v.norm()); geo["frac_norm_perp_probe"] = float(v_perp.norm()/opt_v.norm())
um = md/md.norm(); geo["frac_norm_parallel_meandiff"] = float(torch.dot(opt_v,um).abs()/opt_v.norm())
print("GEOMETRY:", json.dumps(geo, indent=1), flush=True)
v_par_s = v_par / v_par.norm() * opt_v.norm(); v_perp_s = v_perp / v_perp.norm() * opt_v.norm()
recs = []
def gen(rows, mask=None):
    pr = prompts(rows); outs = [None]*len(rows)
    groups = {True: [], False: []}
    for i in range(len(rows)): groups[True if mask is None else bool(mask[i])].append(i)
    sL, sV = state["L"], state["vec"]
    for on, idxs in groups.items():
        if not idxs: continue
        state["L"] = sL if on else None; state["vec"] = sV
        for j in range(0, len(idxs), 8):
            sub = idxs[j:j+8]; inp = tok([pr[k] for k in sub], return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad(): o = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
            for k,row in zip(sub, o[:, inp.input_ids.shape[1]:]):
                outs[k] = re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip()
    state["L"], state["vec"] = sL, sV
    return outs
def emit(cond, resp):
    for r,x in zip(test, resp): recs.append({"condition":cond,"id":r["id"],"label":r["label"],"entity":r.get("entity",""),"response":x})
arms = [("v_par_uncond", v_par_s, None), ("v_par_cast", v_par_s, gate), ("v_perp_uncond", v_perp_s, None), ("v_perp_cast", v_perp_s, gate),
        ("neg_v_uncond", -opt_v, None), ("neg_v_half_uncond", -0.5*opt_v, None), ("v_cast_rerun", opt_v, gate), ("base_rerun", None, None)]
for name, vec, mask in arms:
    state["L"] = None if vec is None else L_V; state["vec"] = None if vec is None else vec.to("cuda:0")
    emit(name, gen(test, mask)); print("arm done:", name, flush=True)
state["L"] = None
with open(f"{D}/steer_geometry_gens.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
json.dump({"geometry": geo, "gate_threshold": th, "gate_test_on_fic": float(gate[yte==1].mean()), "gate_test_on_real": float(gate[yte==0].mean()),
           "per_item_gate": [{"id": r["id"], "label": r["label"], "probe_score": float(s), "gate_on": bool(g)} for r,s,g in zip(test, tes, gate)]},
          open(f"{D}/steer_geometry.json","w"), indent=1)
print("WROTE steer_geometry.json / steer_geometry_gens.jsonl", flush=True)
