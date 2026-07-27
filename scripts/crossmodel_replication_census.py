#!/usr/bin/env python3
"""CROSS-MODEL REPLICATION on Llama-3.1-8B-Instruct of two Qwen3-only headline results:
 (a) domain-general knowledge-gap transfer (4x4 matrix over {scm,biomed,people,academic})
 (b) familiarity tier: post-cutoff-real position on the well-known-vs-fictional axis.

Llama mid layer = hidden_states index 16 (of 33; = layer 16 of 32). Mean-pooled prompt state.
Leak cleaning uses crossdomain_leak_verdicts_census.jsonl: drop pairs whose fictional entity verdict=='leaked'.
Probe: StandardScaler + LogReg(C=0.3), matching the Qwen scripts."""
import os
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

P = os.environ.get("HN_DATA", "./data")
L = 16  # Llama mid layer (of 32)

# ---- leak verdicts: pairs to DROP ----
leaked_grps = set()
for line in open(f"{P}/crossdomain_leak_verdicts_census.jsonl"):
    r = json.loads(line)
    if str(r.get("verdict", "")).lower() == "leaked":
        leaked_grps.add(r["grp"])

# ---- load Llama reps ----
inst = np.load(f"{P}/repr/llama31_instruct.npz", allow_pickle=True)   # SCM parametric arm here
xd = np.load(f"{P}/repr/llama31_crossdomain.npz", allow_pickle=True)  # biomed/people/academic
tmp = np.load(f"{P}/repr/llama31_temporal.npz", allow_pickle=True)    # temporal tiers

# SCM domain = parametric arm of the main instruct extraction (matches Qwen's load_scm)
ms = inst["arm"] == "parametric"
Xs = inst["meanpool"][L][ms].astype(np.float32)
ys = inst["y"][ms].astype(int)
gs = inst["grp"][ms]

Xn = xd["meanpool"][L].astype(np.float32)
domn, yn, grpn = xd["dom"], xd["y"].astype(int), xd["grp"]

DOMAINS = ["scm", "biomed", "people", "academic"]
data = {}
data["scm"] = dict(X=Xs, y=ys, g=gs, dropped=0, total=len(np.unique(gs)))
for dom in ["biomed", "people", "academic"]:
    mask = domn == dom
    Xd, yd, gd = Xn[mask], yn[mask], grpn[mask]
    keep = np.array([g not in leaked_grps for g in gd])
    n_drop = len(set(g for g, k in zip(gd, keep) if not k))
    data[dom] = dict(X=Xd[keep], y=yd[keep], g=gd[keep], dropped=n_drop, total=len(np.unique(gd)))

def within(d):
    X, y, g = d["X"], d["y"], d["g"]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)

def cross(dtr, dte):
    sc = StandardScaler().fit(dtr["X"])
    clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(dtr["X"]), dtr["y"])
    return roc_auc_score(dte["y"], clf.predict_proba(sc.transform(dte["X"]))[:, 1])

M = {a: {b: (within(data[a]) if a == b else cross(data[a], data[b])) for b in DOMAINS} for a in DOMAINS}
diag = [M[d][d] for d in DOMAINS]
off = [M[a][b] for a in DOMAINS for b in DOMAINS if a != b]

lodo = {}
for held in DOMAINS:
    tr = [d for d in DOMAINS if d != held]
    Xtr = np.concatenate([data[d]["X"] for d in tr]); ytr = np.concatenate([data[d]["y"] for d in tr])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
    lodo[held] = roc_auc_score(data[held]["y"], clf.predict_proba(sc.transform(data[held]["X"]))[:, 1])

counts = {d: {"pairs_used": int(len(np.unique(data[d]["g"]))),
              "leak_dropped_pairs": int(data[d]["dropped"]),
              "pairs_generated": int(data[d]["total"])} for d in DOMAINS}

# ---------- (b) familiarity tier ----------
mp = inst["arm"] == "parametric"
Xtr = inst["meanpool"][L, mp, :].astype(np.float32); ytr = inst["y"][mp].astype(int)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(Xtr), ytr)
tier = tmp["tier"]; Xt = tmp["meanpool"][L, :, :].astype(np.float32)
s = clf.predict_proba(sc.transform(Xt))[:, 1]
tiers = {}
for t in ["wellknown", "postcutoff", "fictional"]:
    v = s[tier == t]
    tiers[t] = {"mean": float(v.mean()), "median": float(np.median(v)),
                "std": float(v.std()), "n": int(len(v))}
wk, poc, fic = tiers["wellknown"]["mean"], tiers["postcutoff"]["mean"], tiers["fictional"]["mean"]
frac = float((poc - wk) / (fic - wk + 1e-9))

out = {
    "model": "meta-llama-Llama-3.1-8B-Instruct", "layer_index": L, "layer_of": "16 of 32",
    "probe": "StandardScaler+LogReg(C=0.3)", "leak_cleaning": "crossdomain_leak_verdicts_census.jsonl verdict=='leaked' pairs dropped",
    "n_leaked_pairs_total": len(leaked_grps),
    "transfer": {
        "matrix_train_rows_test_cols": M,
        "diagonal_within_domain": {d: M[d][d] for d in DOMAINS},
        "mean_within": float(np.mean(diag)),
        "mean_cross_offdiag": float(np.mean(off)),
        "lodo_train3_test_heldout": lodo,
        "mean_lodo": float(np.mean(list(lodo.values()))),
        "counts": counts,
    },
    "familiarity": {
        "tiers": tiers,
        "postcutoff_fraction_toward_fictional": frac,
        "note": "Llama-3.1 documented training cutoff Dec 2023; post-2025 entities are post-cutoff.",
    },
}

# verdicts
tv_a = ("DOMAIN-GENERAL: mean cross-domain AUROC=%.3f (within=%.3f, LODO=%.3f) -- %s"
        % (out["transfer"]["mean_cross_offdiag"], out["transfer"]["mean_within"], out["transfer"]["mean_lodo"],
           ("domain-general on Llama (>=0.9 cross)" if out["transfer"]["mean_cross_offdiag"] >= 0.9
            else "transfers but below 0.9 cross" if out["transfer"]["mean_cross_offdiag"] >= 0.75
            else "weak/does-not-transfer on Llama")))
if fic > wk:
    loc = ("intermediate/fictional-ward (frac=%.0f%%)" % (frac * 100) if 0.15 < frac < 0.85
           else "well-known-ward (frac=%.0f%%)" % (frac * 100) if frac <= 0.15
           else "fictional-ward (frac=%.0f%%)" % (frac * 100))
else:
    loc = "axis degenerate (fictional<=wellknown)"
tv_b = ("FAMILIARITY: post-cutoff P(fic) mean=%.3f between wellknown=%.3f and fictional=%.3f -> %s"
        % (poc, wk, fic, loc))
out["verdicts"] = {"transfer": tv_a, "familiarity": tv_b}

json.dump(out, open(f"{P}/crossmodel_replication_census.json", "w"), indent=2)

print("=== TRANSFER MATRIX (rows=train, cols=test) AUROC — Llama L16 ===")
print(f"{'':10s}" + "".join(f"{b:>11s}" for b in DOMAINS))
for a in DOMAINS:
    print(f"{a:10s}" + "".join(f"{M[a][b]:>11.3f}" for b in DOMAINS))
print(f"\nmean within={np.mean(diag):.3f}  mean cross={np.mean(off):.3f}  mean LODO={np.mean(list(lodo.values())):.3f}")
print("counts:", json.dumps(counts))
print("\n=== FAMILIARITY TIERS (P fictional-like) ===")
for t in ["wellknown", "postcutoff", "fictional"]:
    print(f"  {t:11s} mean={tiers[t]['mean']:.3f} median={tiers[t]['median']:.3f} n={tiers[t]['n']}")
print(f"  postcutoff fraction toward fictional: {frac:.0%}")
print("\nVERDICTS:\n ", tv_a, "\n ", tv_b)
print("wrote crossmodel_replication_census.json")
