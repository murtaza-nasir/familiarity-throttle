#!/usr/bin/env python3
"""TMLR revision C1: model-generic component-level localization of the familiarity->abstention readout.
Same design as component_throttle.py / component_throttle_llama.py (AtP screen over every attention head and MLP block,
true-patching verification of top-20 + 20 random controls, joint top-k and FULL patch sanity, entity-span attention mass),
generalized to: multimodal wrappers (decoder layers found by search), per-layer head_dim (Gemma 4), hybrid layers without
self_attn (Qwen3.5 linear-attention blocks are one component each), device_map across visible GPUs.
Env: MODEL (hf id or path), TAG (output suffix), SMOKE=1 (4 pairs), N_PAIRS (default 96), THINK_KW=0 to skip enable_thinking kw."""
import json, os, gc, random, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
P = "$HN_ROOT"; D = f"{P}/data/abstention_full"
MODEL = os.environ["MODEL"]; TAG = os.environ["TAG"]; SMOKE = os.environ.get("SMOKE") == "1"
OUT = f"{D}/component_throttle_{TAG}{'_smoke' if SMOKE else ''}.json"
SEED = 0; N_PAIRS = 4 if SMOKE else int(os.environ.get("N_PAIRS", "96")); TOPK_VERIFY = 20; N_CONTROLS = 20
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
z = np.load(f"{D}/repr/qwen3_instruct.npz", allow_pickle=True)
pm = z["arm"] == "parametric"; grp = z["grp"][pm].astype(str); pairs = sorted(set(grp))
rng = np.random.RandomState(SEED); rng.shuffle(pairs); half = len(pairs)//2
test_pairs = sorted(pairs[half:]); use_pairs = test_pairs[:N_PAIRS]
print(f"total pairs={len(pairs)} test={len(test_pairs)} using {len(use_pairs)}", flush=True)
items = [json.loads(l) for l in open(f"{D}/items.jsonl")]; by_id = {it["id"]: it for it in items}
tok = AutoTokenizer.from_pretrained(MODEL)
def build_prompt(q):
    msgs = [{"role": "user", "content": q}]
    if os.environ.get("THINK_KW", "1") == "1":
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception: pass
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("PROMPT EXAMPLE:", repr(build_prompt("What is X?")), flush=True)
pair_data = []
for g in use_pairs:
    it_r, it_f = by_id[f"{g}_ans"], by_id[f"{g}_abs"]
    pr, pf = build_prompt(it_r["question"]), build_prompt(it_f["question"])
    ids_r = tok(pr, add_special_tokens=False)["input_ids"]; ids_f = tok(pf, add_special_tokens=False)["input_ids"]
    S = 0
    while S < min(len(ids_r), len(ids_f)) and ids_r[-1-S] == ids_f[-1-S]: S += 1
    assert 1 <= S < min(len(ids_r), len(ids_f)), (g, S)
    pair_data.append(dict(g=g, prompt_r=pr, prompt_f=pf, ids_r=ids_r, ids_f=ids_f, S=S, ent_r=it_r["entity"], ent_f=it_f["entity"]))
S_arr = np.array([p["S"] for p in pair_data]); print(f"common-suffix S: min={S_arr.min()} med={np.median(S_arr):.0f} max={S_arr.max()}", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager").eval()
for prm in model.parameters(): prm.requires_grad_(False)
# ---- locate decoder layers generically ----
cands = [(n, m) for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) > 8 and hasattr(m[0], "mlp")]
LAYERS_NAME, LAYERS = cands[0]; NB = len(LAYERS)
embed = model.get_input_embeddings(); dev0 = embed.weight.device
print(f"layers at {LAYERS_NAME}: {NB}; embed device {dev0}", flush=True)
# ---- per-layer attention structure ----
attn_kind, NHb, HDb, attn_out_mod = [], [], [], []
for b in range(NB):
    L = LAYERS[b]
    if hasattr(L, "self_attn"):
        sa = L.self_attn; hd = getattr(sa, "head_dim", None)
        if hd is None:
            cfg = model.config; t = getattr(cfg, "text_config", cfg); hd = getattr(t, "head_dim", t.hidden_size // t.num_attention_heads)
        nh = sa.o_proj.in_features // hd; assert nh * hd == sa.o_proj.in_features, (b, nh, hd, sa.o_proj.in_features)
        attn_kind.append("heads"); NHb.append(nh); HDb.append(hd); attn_out_mod.append(sa.o_proj)
    else:
        alt = getattr(L, "linear_attn", None) or getattr(L, "attn", None)
        assert alt is not None, f"layer {b} has no attention module: {[k for k,_ in L.named_children()]}"
        lins = [m for _, m in alt.named_modules() if isinstance(m, torch.nn.Linear)]
        outp = getattr(alt, "out_proj", None) or getattr(alt, "o_proj", None) or lins[-1]
        attn_kind.append("block"); NHb.append(1); HDb.append(outp.in_features); attn_out_mod.append(outp)
n_comp = sum(NHb) + NB
print(f"components: heads/blocks {sum(NHb)} + MLP {NB} = {n_comp}; kinds {dict((k, attn_kind.count(k)) for k in set(attn_kind))}; head_dims {sorted(set(HDb))}", flush=True)
ABSTAIN_PHRASES = ["Unfortunately", "Sorry", "There is no", "I don't have reliable information"]
COMMIT_PHRASES = ["The", "According to", "Yes", "It is"]
def first_ids(phr):
    out = []
    for p in phr:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if ids: out.append(ids[0])
    return out
ab_ids = first_ids(ABSTAIN_PHRASES); co_ids = first_ids(COMMIT_PHRASES); inter = set(ab_ids) & set(co_ids)
ab_ids = sorted(set(ab_ids) - inter); co_ids = sorted(set(co_ids) - inter)
print("abstain:", [tok.decode([i]) for i in ab_ids], "commit:", [tok.decode([i]) for i in co_ids], flush=True)
assert len(ab_ids) >= 2 and len(co_ids) >= 2
def metric_from_logits(logits):
    lg = logits[:, -1, :].float(); return (lg[:, ab_ids].mean(1) - lg[:, co_ids].mean(1))[0].item()
ST = {"mode": None, "S": None, "real_attn": {}, "real_mlp": {}, "fic_attn": {}, "fic_mlp": {}, "grad_attn": {}, "grad_mlp": {}, "patch_attn": {}, "patch_mlp": set()}
def mk_attn_pre(b):
    HD = HDb[b]
    def pre(mod, args):
        x = args[0]; S = ST["S"]
        if ST["mode"] == "cap_real": ST["real_attn"][b] = x[0, -S:, :].detach().float().cpu()
        elif ST["mode"] == "grad":
            ST["fic_attn"][b] = x[0, -S:, :].detach().float().cpu()
            def gh(g, b=b): ST["grad_attn"][b] = g[0, -S:, :].detach().float().cpu()
            x.register_hook(gh)
        elif ST["mode"] == "patch":
            heads = ST["patch_attn"].get(b)
            if heads:
                x = x.clone(); r = ST["real_attn"][b].to(x.device, x.dtype)
                for h in heads:
                    if h == "all": x[0, -S:, :] = r
                    else: x[0, -S:, h*HD:(h+1)*HD] = r[:, h*HD:(h+1)*HD]
                return (x,) + tuple(args[1:])
        return None
    return pre
def mk_mlp(b):
    def hook(mod, inp, out):
        S = ST["S"]; o = out[0] if isinstance(out, tuple) else out
        if ST["mode"] == "cap_real": ST["real_mlp"][b] = o[0, -S:, :].detach().float().cpu()
        elif ST["mode"] == "grad":
            ST["fic_mlp"][b] = o[0, -S:, :].detach().float().cpu()
            def gh(g, b=b): ST["grad_mlp"][b] = g[0, -S:, :].detach().float().cpu()
            o.register_hook(gh)
        elif ST["mode"] == "patch" and b in ST["patch_mlp"]:
            o = o.clone(); o[0, -S:, :] = ST["real_mlp"][b].to(o.device, o.dtype)
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return None
    return hook
for b in range(NB):
    attn_out_mod[b].register_forward_pre_hook(mk_attn_pre(b)); LAYERS[b].mlp.register_forward_hook(mk_mlp(b))
def run(ids, mode, grad=False):
    ST["mode"] = mode; t = torch.tensor([ids], device=dev0)
    if grad:
        emb = embed(t).detach().requires_grad_(True); return model(inputs_embeds=emb).logits
    with torch.no_grad(): return model(input_ids=t).logits
n = len(pair_data)
attr_attn = [np.zeros((n, NHb[b])) for b in range(NB)]; attr_mlp = np.zeros((n, NB)); m_fic = np.zeros(n); m_real = np.zeros(n)
print("PHASE 1: attribution patching", flush=True)
for i, pd in enumerate(pair_data):
    ST["S"] = pd["S"]
    for k in ["real_attn","real_mlp","fic_attn","fic_mlp","grad_attn","grad_mlp"]: ST[k].clear()
    m_real[i] = metric_from_logits(run(pd["ids_r"], "cap_real"))
    logits = run(pd["ids_f"], "grad", grad=True); lg = logits[:, -1, :].float()
    m = lg[:, ab_ids].mean(1) - lg[:, co_ids].mean(1); m_fic[i] = m[0].item(); m.sum().backward()
    for b in range(NB):
        da = ST["real_attn"][b] - ST["fic_attn"][b]
        attr_attn[b][i] = (da * ST["grad_attn"][b]).view(pd["S"], NHb[b], HDb[b]).sum(dim=(0, 2)).numpy()
        dm = ST["real_mlp"][b] - ST["fic_mlp"][b]; attr_mlp[i, b] = float((dm * ST["grad_mlp"][b]).sum())
    if (i+1) % 16 == 0: print(f"  pair {i+1}/{n}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
gap = m_fic - m_real
print(f"baseline: m_fic={m_fic.mean():+.3f} m_real={m_real.mean():+.3f} gap={gap.mean():+.3f}", flush=True)
comps = []
for b in range(NB):
    if attn_kind[b] == "heads":
        for h in range(NHb[b]): comps.append((f"L{b}.H{h}", b, h, float(attr_attn[b][:, h].mean())))
    else: comps.append((f"L{b}.LIN", b, 0, float(attr_attn[b][:, 0].mean())))
    comps.append((f"L{b}.MLP", b, None, float(attr_mlp[:, b].mean())))
comps_ranked = sorted(comps, key=lambda c: -abs(c[3])); top = comps_ranked[:TOPK_VERIFY]; top_names = {c[0] for c in top}
pool = comps_ranked[TOPK_VERIFY:]; ctrl_rng = np.random.RandomState(SEED)
controls = [pool[j] for j in ctrl_rng.choice(len(pool), size=N_CONTROLS, replace=False)]
print("top-20 by |AtP|:", [(c[0], round(c[3], 3)) for c in top], flush=True)
def set_patch(comp_list):
    ST["patch_attn"] = {}; ST["patch_mlp"] = set()
    for (name, b, h, _) in comp_list:
        if h is None: ST["patch_mlp"].add(b)
        else: ST["patch_attn"].setdefault(b, []).append(h)
verify_list = top + controls; true_eff = np.zeros((len(verify_list), n)); KS = [1, 5, 10, 20, 50]
joint_eff = np.zeros((len(KS), n)); full_eff = np.zeros(n)
print("PHASE 2: true patching", flush=True)
for i, pd in enumerate(pair_data):
    ST["S"] = pd["S"]; ST["real_attn"].clear(); ST["real_mlp"].clear(); _ = run(pd["ids_r"], "cap_real")
    for ci, comp in enumerate(verify_list):
        set_patch([comp]); true_eff[ci, i] = metric_from_logits(run(pd["ids_f"], "patch")) - m_fic[i]
    for ki, k in enumerate(KS):
        set_patch(comps_ranked[:k]); joint_eff[ki, i] = metric_from_logits(run(pd["ids_f"], "patch")) - m_fic[i]
    ST["patch_attn"] = {b: ["all"] for b in range(NB)}; ST["patch_mlp"] = set(range(NB))
    full_eff[i] = metric_from_logits(run(pd["ids_f"], "patch")) - m_fic[i]
    if (i+1) % 16 == 0: print(f"  pair {i+1}/{n}", flush=True)
ST["patch_attn"] = {}; ST["patch_mlp"] = set()
full_resid = full_eff - (m_real - m_fic)
print(f"FULL patch max |resid| = {np.abs(full_resid).max():.4f}", flush=True)
if SMOKE:
    assert np.abs(full_resid).max() < 0.05, "FULL patch failed to reproduce m_real"; print("SMOKE OK", flush=True)
def boot_ci(v, B=1000, seed=0):
    r = np.random.RandomState(seed); idx = r.randint(0, len(v), size=(B, len(v))); bm = v[idx].mean(1)
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
verified = []
for ci, (name, b, h, atp) in enumerate(verify_list):
    v = true_eff[ci]; lo, hi = boot_ci(v, seed=SEED+ci)
    verified.append({"component": name, "block": b, "head": h, "role": "top" if name in top_names else "control", "atp_mean": atp,
                     "true_effect_mean": float(v.mean()), "ci_low": lo, "ci_high": hi, "frac_gap_closed": float(-v.mean()/gap.mean()), "ci_excludes_0": bool(lo > 0 or hi < 0)})
full_mean = float(full_eff.mean()); concentration = []
for ki, k in enumerate(KS):
    v = joint_eff[ki]; lo, hi = boot_ci(v, seed=777+ki)
    concentration.append({"k": k, "joint_effect_mean": float(v.mean()), "ci_low": lo, "ci_high": hi, "frac_of_full": float(v.mean()/full_mean), "frac_gap_closed": float(-v.mean()/gap.mean())})
    print(f"  top-{k} joint: {v.mean():+.4f} frac_of_full={v.mean()/full_mean:.3f}", flush=True)
# PHASE 3 attention patterns (heads only; robust to hybrid models whose attentions tuple has gaps)
top_heads = [(name, b, h) for (name, b, h, _) in top if h is not None and attn_kind[b] == "heads"]
attn_summary = []; uniform_expect = None
try:
    if top_heads:
        ent_mass = {t[0]: [] for t in top_heads}; ent_all = []; spanfr = []
        for pd in pair_data:
            enc = tok(pd["prompt_f"], add_special_tokens=False, return_offsets_mapping=True)
            if enc["input_ids"] != pd["ids_f"]: continue
            pos0 = pd["prompt_f"].find(pd["ent_f"])
            if pos0 < 0: continue
            pos1 = pos0 + len(pd["ent_f"]); span = [ti for ti, (a, bb) in enumerate(enc["offset_mapping"]) if a < pos1 and bb > pos0]
            if not span: continue
            ST["mode"] = None
            with torch.no_grad(): out = model(input_ids=torch.tensor([pd["ids_f"]], device=dev0), output_attentions=True)
            atts = out.attentions
            if atts is None: break
            # map attentions to blocks: if one entry per block use index; else assume entries correspond to heads-kind blocks in order
            heads_blocks = [b for b in range(NB) if attn_kind[b] == "heads"]
            if len(atts) == NB: amap = {b: atts[b] for b in heads_blocks}
            elif len(atts) == len(heads_blocks): amap = {b: atts[i] for i, b in enumerate(heads_blocks)}
            else: amap = {b: atts[b] for b in heads_blocks if b < len(atts)}
            per = {}
            for b, A in amap.items():
                if A is None: continue
                row = A[0, :, -1, :].float().cpu().numpy(); per[b] = row[:, span].sum(1)
            if not per: break
            ent_all.append(per)
            for (name, b, h) in top_heads:
                if b in per: ent_mass[name].append(float(per[b][h]))
            spanfr.append(len(span)/len(pd["ids_f"])); del out; gc.collect(); torch.cuda.empty_cache()
        if ent_all:
            blocks_ok = [b for b in ent_all[0] if all(b in e for e in ent_all)]
            allv = np.concatenate([np.mean([e[b] for e in ent_all], axis=0) for b in blocks_ok]); flat = np.sort(allv)
            for (name, b, h) in top_heads:
                if not ent_mass[name]: continue
                mval = float(np.mean(ent_mass[name])); attn_summary.append({"component": name, "mean_entity_mass_from_final_pos": mval, "percentile_among_all_heads": float((flat < mval).mean()*100)})
            uniform_expect = float(np.mean(spanfr))
            for a in attn_summary: print(f"  {a['component']}: entity mass={a['mean_entity_mass_from_final_pos']:.3f} pct={a['percentile_among_all_heads']:.1f}", flush=True)
except Exception as e:
    print("PHASE 3 skipped:", repr(e), flush=True); attn_summary = [{"error": repr(e)}]
from collections import Counter
layer_hist = Counter(c[1] for c in top); kind_hist = Counter("mlp" if c[2] is None else ("head" if attn_kind[c[1]] == "heads" else "linblock") for c in top)
top5 = sorted([v for v in verified if v["role"] == "top"], key=lambda v: v["true_effect_mean"])[:5]
out = {"method": {"model": MODEL, "pairs": "held-out TEST half of throttle_trace_v2 seed-0 pair split; first 96 sorted (same ids as Qwen/Llama runs)", "pair_ids": use_pairs,
                  "layers_module": LAYERS_NAME, "n_blocks": NB, "attn_kinds": attn_kind, "heads_per_block": NHb, "head_dim_per_block": HDb, "n_components": n_comp,
                  "S_stats": {"min": int(S_arr.min()), "median": float(np.median(S_arr)), "max": int(S_arr.max())},
                  "sign_convention": "patch REAL activations into FICTIONAL run; negative effect = component carries familiarity readout",
                  "metric": "mean logit(abstain first-tokens) - mean logit(commit first-tokens) at final position",
                  "abstain_tokens": [tok.decode([i]) for i in ab_ids], "commit_tokens": [tok.decode([i]) for i in co_ids],
                  "atp": "attr = sum_pos (act_real - act_fic) . grad_fic(metric)", "verification": f"true patching of top-{TOPK_VERIFY} + {N_CONTROLS} random controls, bootstrap B=1000"},
       "baseline": {"m_fictional_mean": float(m_fic.mean()), "m_real_mean": float(m_real.mean()), "gap_mean": float(gap.mean()), "gap_ci": list(boot_ci(gap, seed=1234))},
       "atp_top50": [{"component": c[0], "block": c[1], "head": c[2], "atp_mean": c[3]} for c in comps_ranked[:50]],
       "verified_effects": verified, "full_patch": {"effect_mean": full_mean, "expected_minus_gap": float(-gap.mean()), "max_perpair_residual": float(np.abs(full_resid).max())},
       "concentration_true_patching": concentration, "layer_hist_top20": {str(k): v for k, v in sorted(layer_hist.items())}, "kind_top20": dict(kind_hist),
       "attention_patterns": {"top_heads": attn_summary, "uniform_expectation_spanfrac": uniform_expect},
       "summary": {"top5": [{"component": v["component"], "true_effect_mean": v["true_effect_mean"], "ci": [v["ci_low"], v["ci_high"]], "frac_gap_closed": v["frac_gap_closed"]} for v in top5],
                   "frac_gap_closed_joint": {str(c["k"]): c["frac_gap_closed"] for c in concentration}},
       "seed": SEED, "n_pairs": n, "smoke": SMOKE}
json.dump(out, open(OUT, "w"), indent=2); print("wrote", OUT, flush=True)
print("SUMMARY:", json.dumps(out["summary"], indent=1), flush=True)
