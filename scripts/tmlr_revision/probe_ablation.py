#!/usr/bin/env python3
"""TMLR revision C7: probe-construction ablation from cached mean-pooled reps (CPU).
In-domain (Qwen3-8B parametric arm, GroupKFold(5) by pair, OOF AUROC): layer sweep x C grid, with/without StandardScaler.
Cross-domain (train supply-chain parametric -> test crossdomain_new.npz all domains; leave-one-domain-out): layer x C at the paper's protocol.
Also Llama-3.1-8B in-domain layer sweep (mean-pool as stored). Writes probe_ablation.json"""
import json, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = "$HN_ROOT/data/abstention_full"
def load(f):
    z = np.load(f"{D}/repr/{f}", allow_pickle=True); return z, {k: z[k].shape for k in z.files}
z, sh = load("qwen3_instruct.npz"); print("qwen3_instruct", sh, flush=True)
pm = z["arm"] == "parametric"; y = z["y"][pm].astype(int); grp = z["grp"][pm].astype(str); mp = z["meanpool"][:, pm, :].astype(np.float32)
nL = mp.shape[0]
def oof_auroc(X, y, grp, C=0.3, scale=True):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups=grp):
        Xtr, Xte = X[tr], X[te]
        if scale: sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr]); oof[te] = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y, oof))
res = {"in_domain_qwen3_8b": {"layers": {}, "C_grid_at_L18": {}, "no_scaler_L18": None}}
for L in range(0, nL, 1):
    res["in_domain_qwen3_8b"]["layers"][str(L)] = oof_auroc(mp[L], y, grp)
    print(f"L{L} {res['in_domain_qwen3_8b']['layers'][str(L)]:.4f}", flush=True)
for C in [0.001, 0.01, 0.1, 0.3, 1.0, 10.0]:
    res["in_domain_qwen3_8b"]["C_grid_at_L18"][str(C)] = oof_auroc(mp[18], y, grp, C=C)
res["in_domain_qwen3_8b"]["no_scaler_L18"] = oof_auroc(mp[18], y, grp, scale=False)
print("C grid:", res["in_domain_qwen3_8b"]["C_grid_at_L18"], "noscale:", res["in_domain_qwen3_8b"]["no_scaler_L18"], flush=True)
# cross-domain
try:
    zc, shc = load("crossdomain_new.npz"); print("crossdomain_new", shc, list(zc.files), flush=True)
    yc = zc["y"].astype(int); dom = zc["domain"].astype(str) if "domain" in zc.files else np.array(["x"]*len(yc)); mc = zc["meanpool"].astype(np.float32)
    print("domains:", {d: int((dom == d).sum()) for d in set(dom)}, flush=True)
    cd = {"layers": {}, "C_at_L18": {}}
    for L in range(0, nL, 2):
        sc = StandardScaler().fit(mp[L]); clf = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(mp[L]), y)
        cd["layers"][str(L)] = float(roc_auc_score(yc, clf.predict_proba(sc.transform(mc[L]))[:, 1]))
        print(f"cross L{L} {cd['layers'][str(L)]:.4f}", flush=True)
    for C in [0.001, 0.01, 0.1, 0.3, 1.0, 10.0]:
        sc = StandardScaler().fit(mp[18]); clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(mp[18]), y)
        cd["C_at_L18"][str(C)] = float(roc_auc_score(yc, clf.predict_proba(sc.transform(mc[18]))[:, 1]))
    res["cross_domain_qwen3_8b_train_scm_test_all"] = cd
except Exception as e:
    res["cross_domain_error"] = str(e); print("cross-domain error", e, flush=True)
try:
    zl, shl = load("llama31_instruct.npz"); print("llama", shl, flush=True)
    pl = zl["arm"] == "parametric"; yl = zl["y"][pl].astype(int); gl = zl["grp"][pl].astype(str); ml = zl["meanpool"][:, pl, :].astype(np.float32)
    res["in_domain_llama31_8b_layers"] = {str(L): oof_auroc(ml[L], yl, gl) for L in range(0, ml.shape[0], 2)}
    print("llama layers:", res["in_domain_llama31_8b_layers"], flush=True)
except Exception as e:
    res["llama_error"] = str(e)
json.dump(res, open(f"{D}/probe_ablation.json", "w"), indent=1); print("wrote probe_ablation.json", flush=True)
