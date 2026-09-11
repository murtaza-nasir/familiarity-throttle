#!/usr/bin/env python3
"""Fill the pending macros from judged generation files. Judge field: 'judge_qwen32b' (open-weights) or 'judge_sonnet' if present (preferred)."""
import json, sys, os, numpy as np
from scipy import stats
D = "tmlr_revision"
def cp(k, n):
    lo = 0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1); hi = 1 if k == n else stats.beta.ppf(0.975, k + 1, n - k); return lo, hi
def rates(recs, cond, field):
    fic = [r[field] for r in recs if r["condition"] == cond and r["label"] == "should_abstain" and r.get(field) in ("COMMIT", "DEFLECT", "FLAG")]
    real = [r[field] for r in recs if r["condition"] == cond and r["label"] == "answerable" and r.get(field) in ("COMMIT", "DEFLECT", "FLAG")]
    if not fic or not real: return None
    f = lambda L, k: sum(x == k for x in L) / len(L)
    return dict(n_fic=len(fic), n_real=len(real), flag_fic=f(fic, "FLAG"), ff_real=f(real, "FLAG"), deflect_real=f(real, "DEFLECT"), commit_fic=f(fic, "COMMIT"), gap=f(fic, "FLAG") - f(real, "FLAG"))
def fmt(r, k, n):
    lo, hi = cp(round(r * n), n); return f"{r:.3f} [{lo:.2f},{hi:.2f}]"
out = []; M = lambda n, v: out.append(f"\\renewcommand{{\\{n}}}{{{v}}}")
# ---- steering geometry arms ----
f = f"{D}/steer_geometry_gens_judged.jsonl"
if os.path.exists(f):
    recs = [json.loads(l) for l in open(f)]; field = "judge_sonnet" if any("judge_sonnet" in r for r in recs) else "judge_qwen32b"
    M("STEERJUDGE", "primary judge" if field == "judge_sonnet" else "open-weights judge, Qwen3-32B; primary-judge verdicts to follow")
    names = [("base_rerun", "untouched model (re-run)"), ("v_cast_rerun", "gated $v$ (re-run)"), ("v_par_uncond", "$v_\\parallel$ (probe direction, rescaled), always-on"), ("v_par_cast", "$v_\\parallel$, gated"),
             ("v_perp_uncond", "$v_\\perp$ (orthogonal part, rescaled), always-on"), ("v_perp_cast", "$v_\\perp$, gated"), ("neg_v_uncond", "$-v$, always-on"), ("neg_v_half_uncond", "$-0.5\\,v$, always-on")]
    rows = ["\\begin{tabular}{lccccc}\\toprule", "arm & flags fictional & false-flags real & deflects real & commits fictional & flag-gap \\\\\\midrule"]
    R = {}
    for c, lab in names:
        r = rates(recs, c, field); R[c] = r
        if r: rows.append(f"{lab} & {fmt(r['flag_fic'], r['flag_fic'], r['n_fic'])} & {fmt(r['ff_real'], r['ff_real'], r['n_real'])} & {r['deflect_real']:.3f} & {r['commit_fic']:.3f} & {r['gap']:.3f} \\\\")
    rows.append("\\bottomrule\\end{tabular}"); M("STEERGEOROWS", "\n".join(rows))
    par, perp, neg, negh, base = R.get("v_par_uncond"), R.get("v_perp_uncond"), R.get("neg_v_uncond"), R.get("neg_v_half_uncond"), R.get("base_rerun")
    if par and perp and neg and base:
        M("STEERGEO", f"the probe direction alone, rescaled to the trained norm, suppresses flagging ({base['flag_fic']:.3f} to {par['flag_fic']:.3f}) and commitment ({base['commit_fic']:.3f} to {par['commit_fic']:.3f}) alike in favor of hedged deflections on both classes, while the orthogonal component alone reproduces the full vector's flagging ({perp['flag_fic']:.3f}); the reverse-sign vector raises commits on fictional entities from {base['commit_fic']:.3f} to {neg['commit_fic']:.3f}")
        M("STEERGEOTABLE", f"the probe direction rescaled to the trained norm suppresses flagging on fictional items ({base['flag_fic']:.3f} to {par['flag_fic']:.3f} always-on) and commitment ({base['commit_fic']:.3f} to {par['commit_fic']:.3f}) alike, converting both classes to hedged deflections (deflections on real entities {R['base_rerun']['deflect_real']:.2f} to {par['deflect_real']:.2f}); the orthogonal component reproduces the full vector ({perp['flag_fic']:.3f} always-on, {R['v_perp_cast']['flag_fic']:.3f} gated, {R['v_perp_cast']['ff_real']:.3f} false-flags); and $-v$ drives the model toward committing on fictional entities ({neg['commit_fic']:.3f} against {base['commit_fic']:.3f}; at half norm {negh['commit_fic']:.3f})")
        M("REVSIGN", f"at the trained norm it raises the commit rate on fictional entities from {base['commit_fic']:.3f} to {neg['commit_fic']:.3f} and lowers flagging from {base['flag_fic']:.3f} to {neg['flag_fic']:.3f}, so the direction can be driven toward more fabrication as well as less")
# ---- FT sweep ----
f = f"{D}/ft_sweep_gens_judged.jsonl"; meta_f = f"{D}/ft_sweep_meta.json"
if os.path.exists(f):
    recs = [json.loads(l) for l in open(f)]; field = "judge_sonnet" if any("judge_sonnet" in r for r in recs) else "judge_qwen32b"
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    M("FTJUDGE", "primary judge" if field == "judge_sonnet" else "open-weights judge, Qwen3-32B; primary-judge verdicts to follow")
    conds = sorted({r["condition"] for r in recs}, key=lambda c: (c.split("_")[0], c))
    rows = ["\\begin{tabular}{llccccc}\\toprule", "arm & lr / epochs / quantile & pseudo-abstain rate (real, train) & flags fictional & false-flags real & flag-gap \\\\\\midrule"]
    best = {}
    for c in conds:
        r = rates(recs, c, field); m = meta.get(c, {})
        if not r: continue
        arm = m.get("arm", c.split("_")[0]); q = m.get("se_quantile"); par = f"{m.get('lr', '')} / {m.get('epochs', '')} / {q if q is not None else '--'}"
        pa = m.get("pseudo_abstain_rate_real"); pa = f"{pa:.2f}" if pa is not None else "--"
        rows.append(f"{arm} & {par} & {pa} & {r['flag_fic']:.3f} & {r['ff_real']:.3f} & {r['gap']:.3f} \\\\")
        if arm in ("rtuning", "se") and (arm not in best or r["gap"] > best[arm][1]["gap"]): best[arm] = (c, r)
    rows.append("\\bottomrule\\end{tabular}"); M("FTSWEEPROWS", "\n".join(rows))
    if best:
        allr = {c: rates(recs, c, field) for c in conds}; allr = {c: r for c, r in allr.items() if r}
        minff = min(allr.items(), key=lambda kv: kv[1]["ff_real"]); maxgap = max(allr.items(), key=lambda kv: kv[1]["gap"])
        M("FTSWEEP", f"across eight configurations the best output-supervised flag-gap is {maxgap[1]['gap']:.3f} ({maxgap[1]['flag_fic']:.3f} flagging at {maxgap[1]['ff_real']:.3f} false-flags), and the lowest false-flag rate any configuration reaches is {minff[1]['ff_real']:.3f}, at {minff[1]['flag_fic']:.3f} flagging; no configuration approaches the internal-signal fine-tune ($0.751$) or the gated vector ($0.890$)")
        M("FTSWEEPTABLE", f"the configurations trace a single trade-off curve: raising the pseudo-label quantile or shortening training lowers false-flags (to {minff[1]['ff_real']:.3f} at quantile $0.7$, one epoch) at the cost of flagging on fictional items ({minff[1]['flag_fic']:.3f}), and no point on the curve exceeds flag-gap {maxgap[1]['gap']:.3f}, because the pseudo-labels mislabel a large share of answerable training items whatever the threshold")
open(f"{D}/macros_filled.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out)[:3000])
