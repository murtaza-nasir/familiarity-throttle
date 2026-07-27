#!/usr/bin/env python3
"""WS-0.1 rigor: bootstrap 95% CIs + multi-seed spread on headline AUROCs. Group-level
resampling (resample PAIRS, not items) to respect the matched-pair dependence. Reads
repr/*.npz (cross-model) at each model's best (pos,layer) from probe_layers_summary.json."""
import glob, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

P = os.environ.get("HN_ROOT", ".")
RD = f"{P}/data/repr"
summary = json.load(open(f"{RD}/probe_layers_summary.json"))
B = 1000
RNG = np.random.RandomState(0)


def oof(X, y, grp, seed):
    # GroupKFold is deterministic; emulate seed by permuting group order
    uniq = np.array(sorted(set(grp))); RNG2 = np.random.RandomState(seed); RNG2.shuffle(uniq)
    remap = {g: i for i, g in enumerate(uniq)}; order = np.array([remap[g] for g in grp])
    gkf = GroupKFold(5); o = np.zeros(len(y))
    for tr, te in gkf.split(X, y, order):
        sc = StandardScaler().fit(X[tr]); c = LogisticRegression(C=0.3, solver="liblinear", max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return o


def group_bootstrap_ci(y, s, grp):
    uniq = np.array(sorted(set(grp))); idx_by_g = {g: np.where(grp == g)[0] for g in uniq}
    aucs = []
    for _ in range(B):
        gs = RNG.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[ii])) < 2:
            continue
        aucs.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


out = {}
for f in sorted(glob.glob(f"{RD}/*.npz")):
    tag = os.path.basename(f).replace(".npz", "")
    if tag not in summary:
        continue
    z = np.load(f, allow_pickle=True)
    out[tag] = {}
    for arm in ["parametric", "grounded"]:
        rec = summary[tag][f"{arm}/lasttok"]
        rec2 = summary[tag][f"{arm}/meanpool"]
        pos, L = ("lasttok", rec["best_layer"]) if (rec["best_auroc"] or 0) >= (rec2["best_auroc"] or 0) else ("meanpool", rec2["best_layer"])
        m = z["arm"] == arm; y = z["y"][m]; grp = z["grp"][m]; X = z[pos][L, m, :].astype(np.float32)
        seeds = [oof(X, y, grp, sd) for sd in range(5)]
        pts = [roc_auc_score(y, s) for s in seeds]
        lo, hi = group_bootstrap_ci(y, seeds[0], grp)
        out[tag][arm] = {"pos": pos, "layer": int(L), "auroc": float(np.mean(pts)),
                         "seed_min": float(min(pts)), "seed_max": float(max(pts)), "ci95": [lo, hi], "n": int(len(y))}
        print(f"  {tag:18s} {arm:11s} {pos}@L{L}: AUROC={np.mean(pts):.3f} [seed {min(pts):.3f}-{max(pts):.3f}] CI95=[{lo:.3f},{hi:.3f}] n={len(y)}", flush=True)

json.dump(out, open(f"{P}/data/bootstrap_cis.json", "w"), indent=2)
if "qwen3_base" in out and "qwen3_instruct" in out:
    b = out["qwen3_base"]["parametric"]; i = out["qwen3_instruct"]["parametric"]
    print(f"\nBASE vs INSTRUCT parametric: base {b['auroc']:.3f} CI{b['ci95']} | instruct {i['auroc']:.3f} CI{i['ci95']}")
print("\nwrote bootstrap_cis.json")
