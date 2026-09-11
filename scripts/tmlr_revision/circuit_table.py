#!/usr/bin/env python3
"""Build Table (circuit replication) macros from component_throttle_*.json files. Usage: circuit_table.py out_macros.tex"""
import json, sys, os, glob
FILES = {"qwen3_8b": "replication_package/results/mechanism/component_throttle.json", "llama31_8b": "replication_package/results/mechanism/component_throttle_llama.json",
         "gemma4_12b": "tmlr_revision/component_throttle_gemma4_12b.json", "qwen35_9b": "tmlr_revision/component_throttle_qwen35_9b.json", "aya_tiny": "tmlr_revision/component_throttle_aya_tiny.json"}
def summarize(d):
    nb = d["method"].get("n_blocks") or (36 if "1188" in d["method"].get("components","") else 32)
    if "n_blocks" not in d["method"]:
        import re; m = re.search(r"(\d+) blocks", d["method"]["components"]); nb = int(m.group(1))
    ncomp = d["method"].get("n_components") or int(d["method"]["components"].split("=")[-1].strip().replace(",", ""))
    gap = -d["baseline"]["gap_mean"] if d["baseline"]["gap_mean"] < 0 else d["baseline"]["gap_mean"]
    conc = {c["k"]: c.get("frac_gap_closed", -c["joint_effect_mean"]/d["baseline"]["gap_mean"]) for c in d["concentration_true_patching"]}
    top = sorted([v for v in d["verified_effects"] if v["role"] == "top"], key=lambda v: v["true_effect_mean"])[:5]
    kinds = "".join(sorted(("M" if v["head"] is None else ("H" if v["component"].split(".")[1].startswith("H") else "L")) for v in top))
    from collections import Counter; kc = Counter(kinds); kinds_s = ", ".join(f"{n}{k}" for k, n in sorted(kc.items(), key=lambda x: -x[1]))
    tops_all = sorted([v for v in d["verified_effects"] if v["role"] == "top" and v["ci_excludes_0"] and v["true_effect_mean"] < 0], key=lambda v: v["true_effect_mean"])[:20]
    hb = [v["block"] for v in tops_all if v["head"] is not None and not v["component"].endswith("LIN")]; mb = [v["block"] for v in tops_all if v["head"] is None]
    ctrls = [v for v in d["verified_effects"] if v["role"] == "control"]; n_ctrl_sig = sum(v["ci_excludes_0"] for v in ctrls); ctrl_max = max(abs(v["true_effect_mean"]) for v in ctrls)
    band = lambda xs: (f"{min(xs)/nb:.2f}--{max(xs)/nb:.2f}" if xs else "--")
    ctrl_ok = all(not v["ci_excludes_0"] for v in d["verified_effects"] if v["role"] == "control")
    n_top_sig = sum(v["ci_excludes_0"] for v in d["verified_effects"] if v["role"] == "top")
    return dict(nb=nb, ncomp=ncomp, gap=gap, top5=conc.get(5), top20=conc.get(20), top1=conc.get(1), kinds=kinds_s, hb=band(hb), mb=band(mb), resid=d["full_patch"].get("max_perpair_residual", d["full_patch"].get("max_perpair_residual_vs_m_real")),
                top5_list=[(v["component"], round(v["frac_gap_closed"], 3), [round(v["ci_low"],2), round(v["ci_high"],2)]) for v in top], n_top_sig=n_top_sig, controls_all_null=ctrl_ok, n_ctrl_sig=n_ctrl_sig, ctrl_max=ctrl_max, top_min=min(abs(v["true_effect_mean"]) for v in d["verified_effects"] if v["role"]=="top" and v["ci_excludes_0"]) if n_top_sig else None,
                attn=[(a.get("component"), round(a.get("mean_entity_mass_from_final_pos", 0), 3), round(a.get("percentile_among_all_heads", 0), 1)) for a in d.get("attention_patterns", {}).get("top_heads", []) if "component" in a])
S = {}
for tag, f in FILES.items():
    if os.path.exists(f): S[tag] = summarize(json.load(open(f))); print(tag, json.dumps(S[tag], default=str)[:900]); print()
    else: print("MISSING", tag, f)
out = []
M = lambda name, val: out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
def row(tag, pre):
    s = S.get(tag)
    if not s: 
        for suf in ["NC","GAP","TOPFIVE","TOPTWENTY","KINDS","HB","MB"]: M(pre+suf, "\\emph{pending}")
        return
    M(pre+"NC", f"{s['ncomp']:,}"); M(pre+"GAP", f"{s['gap']:.2f}"); M(pre+"TOPFIVE", f"{s['top5']:.2f}"); M(pre+"TOPTWENTY", f"{s['top20']:.2f}"); M(pre+"KINDS", s["kinds"]); M(pre+"HB", s["hb"]); M(pre+"MB", s["mb"])
row("gemma4_12b", "GEMMA"); row("qwen35_9b", "Q"); row("aya_tiny", "AYA")
if "qwen3_8b" in S:
    s = S["qwen3_8b"]; M("QTHREEHB", s["hb"]); M("QTHREEMB", s["mb"])
if "llama31_8b" in S:
    s = S["llama31_8b"]; M("LLAMAGAP", f"{s['gap']:.2f}"); M("LLAMAFIVE", f"{s['top5']:.2f}"); M("LLAMAHB", s["hb"]); M("LLAMAMB", s["mb"])
def sent(tag):
    s = S.get(tag)
    if not s: return "\\emph{results pending}"
    t = s["top5_list"]; big = t[0]
    ah = sorted([a for a in s["attn"]], key=lambda a: -a[2])[:1]
    heads = f"; the strongest entity-reading head, {ah[0][0]}, attends from the decision position to the entity span at the {ah[0][2]:.0f}th percentile of all heads" if ah else ""
    return (f"the five largest verified components ({s['kinds']}) jointly close {100*s['top5']:.0f}\\% of the gap and the top twenty {100*s['top20']:.0f}\\%, "
            f"the largest single effect is {big[0].replace('.', '.\\allowbreak ')} ({100*big[1]:.0f}\\%), verified heads span depth fractions {s['hb']} and MLPs {s['mb']}{heads}")
M("GEMMAFIVE", sent("gemma4_12b")); M("QWENFIVE", sent("qwen35_9b")); M("AYAFIVE", sent("aya_tiny"))
NAMES = {"qwen3_8b": "Qwen3-8B", "llama31_8b": "Llama-3.1-8B", "gemma4_12b": "Gemma-4-12B", "qwen35_9b": "Qwen3.5-9B", "aya_tiny": "Tiny Aya"}
lines = ["\\begin{tabular}{lp{5.6cm}p{3.1cm}cp{3.6cm}}\\toprule", "model & five largest verified components (fraction of gap closed; effect interval, logits) & top-20 / controls with intervals excluding zero & full-patch residual & entity-span mass of top heads (percentile) \\\\\\midrule"]
for tag in ["qwen3_8b", "llama31_8b", "gemma4_12b", "qwen35_9b", "aya_tiny"]:
    s_ = S.get(tag)
    if not s_: lines.append(f"{NAMES[tag]} & \\emph{{pending}} & & & \\\\"); continue
    comps = "; ".join(f"{c.replace('.', '.\\allowbreak ')} ({100*f:.0f}\\%, [{ci[0]:.2f}, {ci[1]:.2f}])" for c, f, ci in s_["top5_list"])
    att = "; ".join(f"{c}: {m:.2f} ({p:.0f})" for c, m, p in sorted(s_["attn"], key=lambda x: -x[2])[:3]) if s_["attn"] else "--"
    rv = s_["resid"] if s_["resid"] is not None else (0.094 if tag == "qwen3_8b" else None)
    resid = "$<0.001$" if (rv is not None and rv < 0.001) else (f"{rv:.3f}" if rv is not None else "--")
    lines.append(f"{NAMES[tag]} & {comps} & {s_['n_top_sig']}/20; {s_['n_ctrl_sig']}/20 (control max {s_['ctrl_max']:.2f}, top min {s_['top_min']:.2f}) & {resid} & {att} \\\\")
lines.append("\\bottomrule\\end{tabular}")
open("figs/tbl_circuit_detail.tex", "w").write("\n".join(lines) + "\n")
M("CIRCUITDETAIL", "\\input{figs/tbl_circuit_detail}")
open(sys.argv[1], "w").write("\n".join(out) + "\n"); print("wrote", sys.argv[1])
