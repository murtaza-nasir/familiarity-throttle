#!/usr/bin/env python3
"""Cross-domain knowledge-gap transfer matrix. L18 mean-pooled residual.
Domains {scm, biomed, people, academic}. Within-domain GroupKFold(5) by pair id;
cross-domain train-on-all-A test-on-all-B. StandardScaler + LogReg C=0.3."""
import os
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

P = os.environ.get("HN_DATA", "./data")
L = 18
leak = json.load(open(f"{P}/crossdomain_leak_screen_census.json"))
flagged = {d: set(x.lower().strip() for x in v) for d, v in leak.items()}

def load_new():
    d = np.load(f"{P}/repr/crossdomain_new.npz", allow_pickle=True)
    X = d["meanpool"][L].astype(np.float32)  # (N,4096)
    return X, d["dom"], d["y"], d["grp"]

def load_scm():
    d = np.load(f"{P}/repr/qwen3_instruct.npz", allow_pickle=True)
    m = d["arm"] == "parametric"
    return d["meanpool"][L][m].astype(np.float32), d["y"][m], d["grp"][m]

# read entity per grp for new domains to apply leak filter
grp_fic_entity = {}
for fn, dom in [("biomed","biomed"),("people","people"),("academic","academic")]:
    for l in open(f"{P}/crossdomain_{dom}_items.jsonl"):
        r = json.loads(l)
        if r["label"] == "should_abstain":
            grp_fic_entity[r["id"].rsplit("_",1)[0]] = str(r["entity"]).lower().strip()

Xn, domn, yn, grpn = load_new()
Xs, ys, grps = load_scm()

DOMAINS = ["scm", "biomed", "people", "academic"]
data = {}
# scm (no leak screen available for scm fictional -> keep all; note as caveat)
data["scm"] = dict(X=Xs, y=ys.astype(int), g=grps, dropped=0, total=len(np.unique(grps)))
for dom in ["biomed","people","academic"]:
    mask = domn == dom
    Xd, yd, gd = Xn[mask], yn[mask].astype(int), grpn[mask]
    fl = flagged.get(dom, set())
    keep = np.array([grp_fic_entity.get(g, "") not in fl for g in gd])
    n_drop_pairs = len(set(g for g,k in zip(gd,keep) if not k))
    data[dom] = dict(X=Xd[keep], y=yd[keep], g=gd[keep], dropped=n_drop_pairs, total=len(np.unique(gd)))

def within(d):
    X, y, g = d["X"], d["y"], d["g"]
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:,1]
    return roc_auc_score(y, oof)

def cross(dtr, dte):
    sc = StandardScaler().fit(dtr["X"])
    clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(dtr["X"]), dtr["y"])
    p = clf.predict_proba(sc.transform(dte["X"]))[:,1]
    return roc_auc_score(dte["y"], p)

# matrix rows=train, cols=test
M = {}
for a in DOMAINS:
    M[a] = {}
    for b in DOMAINS:
        M[a][b] = within(data[a]) if a==b else cross(data[a], data[b])

diag = [M[d][d] for d in DOMAINS]
off = [M[a][b] for a in DOMAINS for b in DOMAINS if a!=b]

# LODO: train on 3, test on held-out
lodo = {}
for held in DOMAINS:
    tr = [d for d in DOMAINS if d!=held]
    Xtr = np.concatenate([data[d]["X"] for d in tr])
    ytr = np.concatenate([data[d]["y"] for d in tr])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(data[held]["X"]))[:,1]
    lodo[held] = roc_auc_score(data[held]["y"], p)

counts = {d: {"pairs_used": int(len(np.unique(data[d]["g"]))), "leak_dropped_pairs": int(data[d]["dropped"]),
              "pairs_generated": int(data[d]["total"])} for d in DOMAINS}

out = {
    "layer": L, "probe": "StandardScaler+LogReg(C=0.3)",
    "within_domain_cv": "GroupKFold(5) by pair id",
    "cross_domain": "train on all of A, test on all of B",
    "matrix_train_rows_test_cols": M,
    "diagonal_within_domain": {d: M[d][d] for d in DOMAINS},
    "mean_diagonal_within": float(np.mean(diag)),
    "mean_offdiagonal_cross": float(np.mean(off)),
    "lodo_train3_test_heldout": lodo,
    "mean_lodo": float(np.mean(list(lodo.values()))),
    "counts": counts,
    "caveat_leak_screen": {d: len(flagged.get(d,[])) for d in ["biomed","people","academic"]},
    "note": "SCM fictional entities not leak-screened (no screen file); biomed/people/academic exclude GLM-flagged real 'fictional' entities.",
}
json.dump(out, open(f"{P}/crossdomain_transfer.json","w"), indent=2)

print("=== TRANSFER MATRIX (rows=train, cols=test) AUROC ===")
print(f"{'':10s}" + "".join(f"{b:>11s}" for b in DOMAINS))
for a in DOMAINS:
    print(f"{a:10s}" + "".join(f"{M[a][b]:>11.3f}" for b in DOMAINS))
print(f"\nmean diagonal (within-domain): {np.mean(diag):.3f}")
print(f"mean off-diagonal (cross-domain): {np.mean(off):.3f}")
print("\nLODO (train on 3, test held-out):")
for d in DOMAINS: print(f"  {d:10s} {lodo[d]:.3f}")
print(f"  mean LODO: {np.mean(list(lodo.values())):.3f}")
print("\ncounts:", json.dumps(counts))
print(f"wrote {P}/crossdomain_transfer.json")
