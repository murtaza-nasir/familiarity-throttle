#!/usr/bin/env python3
"""Bootstrap CIs (over pairs) for the transfer numbers in Section 5: within-domain (grouped CV OOF), cross-domain
(train all SCM pairs -> each cross-domain arm), leave-one-domain-out; Llama replication; grounding probe -> RAGTruth."""
import json, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = "$HN_ROOT/data/abstention_full"
def boot_auc(y, s, grp, B=2000, seed=0):
    rng = np.random.default_rng(seed); groups = np.unique(grp); idx_by = {g: np.where(grp == g)[0] for g in groups}; out = []
    for _ in range(B):
        pick = rng.choice(groups, len(groups), replace=True); ii = np.concatenate([idx_by[g] for g in pick])
        if len(set(y[ii])) < 2: continue
        out.append(roc_auc_score(y[ii], s[ii]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def fit(X, y): sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(X), y); return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]
res = {}
for tag, f, L in [("qwen3_8b", "qwen3_instruct.npz", 18), ("llama31_8b", "llama31_instruct.npz", 16)]:
    z = np.load(f"{D}/repr/{f}", allow_pickle=True); pm = z["arm"] == "parametric"
    y = z["y"][pm].astype(int); grp = z["grp"][pm].astype(str); X = z["meanpool"][L, pm, :].astype(np.float32)
    if tag == "llama31_8b": X = z["lasttok"][L, pm, :].astype(np.float32)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups=grp): oof[te] = fit(X[tr], y[tr])(X[te])
    r = {"within_scm_oof": float(roc_auc_score(y, oof)), "within_scm_ci": boot_auc(y, oof, grp)}
    cf = "crossdomain_new.npz" if tag == "qwen3_8b" else "llama31_crossdomain.npz"
    zc = np.load(f"{D}/repr/{cf}", allow_pickle=True); print(tag, "crossdomain keys", list(zc.files), flush=True)
    yc = zc["y"].astype(int); gc_ = zc["grp"].astype(str); dom = zc["dom"].astype(str) if "dom" in zc.files else np.array(["all"] * len(yc))
    Xc = (zc["lasttok"] if tag == "llama31_8b" else zc["meanpool"])[L].astype(np.float32)
    leak = json.load(open(f"{D}/crossdomain_leak_screen_census.json")); flagged = {d: set(x.lower().strip() for x in v) for d, v in leak.items()}
    ent = {}
    for dname in ["biomed", "people", "academic"]:
        for l in open(f"{D}/crossdomain_{dname}_items.jsonl"):
            it = json.loads(l); ent[it["id"].rsplit("_", 1)[0]] = ent.get(it["id"].rsplit("_", 1)[0], set()) | {it.get("entity", "").lower().strip()}
    keep = np.array([not any(e in flagged.get(d, set()) for e in ent.get(g, set())) for g, d in zip(gc_, dom)])
    print(tag, "leak-excluded items:", int((~keep).sum()), "pairs:", len(set(gc_[~keep])), flush=True)
    yc, gc_, dom, Xc = yc[keep], gc_[keep], dom[keep], Xc[keep]
    print(tag, "domains", {d: int((dom == d).sum()) for d in set(dom)}, flush=True)
    pred = fit(X, y)
    sc_all = pred(Xc); r["cross_all_from_scm"] = float(roc_auc_score(yc, sc_all)); r["cross_all_ci"] = boot_auc(yc, sc_all, gc_)
    r["cross_by_domain"] = {}
    for d in sorted(set(dom)):
        m = dom == d; r["cross_by_domain"][d] = {"auroc": float(roc_auc_score(yc[m], sc_all[m])), "ci": boot_auc(yc[m], sc_all[m], gc_[m]), "n_items": int(m.sum())}
    # within-domain (grouped CV inside each cross-domain arm) and LODO
    doms = sorted(set(dom)); r["within_by_domain"] = {}; lodo = {}
    for d in doms:
        m = dom == d; oo = np.zeros(m.sum()); Xd, yd, gd = Xc[m], yc[m], gc_[m]
        for tr, te in GroupKFold(5).split(Xd, yd, groups=gd): oo[te] = fit(Xd[tr], yd[tr])(Xd[te])
        r["within_by_domain"][d] = {"auroc": float(roc_auc_score(yd, oo)), "ci": boot_auc(yd, oo, gd)}
        others = ~m; Xtr = np.concatenate([X, Xc[others]]); ytr = np.concatenate([y, yc[others]])
        s = fit(Xtr, ytr)(Xd); lodo[d] = {"auroc": float(roc_auc_score(yd, s)), "ci": boot_auc(yd, s, gd)}
    r["lodo_heldout_domain"] = lodo
    r["mean_within_all4"] = float(np.mean([r["within_scm_oof"]] + [v["auroc"] for v in r["within_by_domain"].values()]))
    res[tag] = r; print(tag, json.dumps({k: v for k, v in r.items() if k in ("within_scm_oof", "within_scm_ci", "cross_all_from_scm", "cross_all_ci")}), flush=True)
# grounding probe -> RAGTruth
try:
    zg = np.load(f"{D}/repr/qwen3_instruct.npz", allow_pickle=True); gm = zg["arm"] == "grounded"
    L = 18; Xg = zg["meanpool"][L, gm, :].astype(np.float32); yg = zg["y"][gm].astype(int); gg = zg["grp"][gm].astype(str)
    oo = np.zeros(len(yg))
    for tr, te in GroupKFold(5).split(Xg, yg, groups=gg): oo[te] = fit(Xg[tr], yg[tr])(Xg[te])
    zr = np.load(f"{D}/repr/ragtruth_qa_verify.npz", allow_pickle=True); print("ragtruth keys", list(zr.files), {k: zr[k].shape for k in zr.files}, flush=True)
    yr = zr["y"].astype(int); Xr = zr["meanpool"][L].astype(np.float32) if zr["meanpool"].ndim == 3 else zr["meanpool"].astype(np.float32)
    gr = zr["grp"].astype(str) if "grp" in zr.files else np.arange(len(yr)).astype(str)
    s = fit(Xg, yg)(Xr)
    oor = np.zeros(len(yr))
    for tr, te in GroupKFold(5).split(Xr, yr, groups=gr): oor[te] = fit(Xr[tr], yr[tr])(Xr[te])
    res["grounding"] = {"within_grounded_oof": float(roc_auc_score(yg, oo)), "within_grounded_ci": boot_auc(yg, oo, gg),
                        "grounded_to_ragtruth": float(roc_auc_score(yr, s)), "grounded_to_ragtruth_ci": boot_auc(yr, s, gr),
                        "ragtruth_within_oof": float(roc_auc_score(yr, oor)), "ragtruth_within_ci": boot_auc(yr, oor, gr), "n_ragtruth": int(len(yr))}
    print("grounding", json.dumps(res["grounding"]), flush=True)
except Exception as e:
    res["grounding_error"] = repr(e); print("grounding error", repr(e), flush=True)
json.dump(res, open(f"{D}/transfer_cis.json", "w"), indent=1); print("wrote transfer_cis.json", flush=True)
