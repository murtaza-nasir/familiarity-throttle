import json, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = "$HN_ROOT/data/abstention_full"
def show(f):
    z = np.load(f"{D}/repr/{f}", allow_pickle=True); print(f, {k: (z[k].shape, str(z[k].dtype)) for k in z.files}, flush=True); return z
zg = show("grounded_verify.npz"); zr = show("ragtruth_qa_verify.npz")
def arr(z):
    keys = [k for k in z.files if z[k].ndim in (2, 3) and z[k].dtype.kind == "f"]; X = z[keys[0]]
    if X.ndim == 3: X = X[18]
    return X.astype(np.float32), keys[0]
def boot_auc(y, s, grp, B=2000, seed=0):
    rng = np.random.default_rng(seed); groups = np.unique(grp); idx_by = {g: np.where(grp == g)[0] for g in groups}; out = []
    for _ in range(B):
        pick = rng.choice(groups, len(groups), replace=True); ii = np.concatenate([idx_by[g] for g in pick])
        if len(set(y[ii])) < 2: continue
        out.append(roc_auc_score(y[ii], s[ii]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def fit(X, y): sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(X), y); return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]
Xg, kg = arr(zg); yg = zg["y"].astype(int); gg = zg["grp"].astype(str) if "grp" in zg.files else np.arange(len(yg)).astype(str)
Xr, kr = arr(zr); yr = zr["y"].astype(int); gr = zr["grp"].astype(str) if "grp" in zr.files else (zr["source_id"].astype(str) if "source_id" in zr.files else np.arange(len(yr)).astype(str))
print("used arrays", kg, kr, "n", len(yg), len(yr), "groups", len(set(gg)), len(set(gr)), flush=True)
oo = np.zeros(len(yg))
for tr, te in GroupKFold(5).split(Xg, yg, groups=gg): oo[te] = fit(Xg[tr], yg[tr])(Xg[te])
oor = np.zeros(len(yr))
for tr, te in GroupKFold(5).split(Xr, yr, groups=gr): oor[te] = fit(Xr[tr], yr[tr])(Xr[te])
s = fit(Xg, yg)(Xr); s2 = fit(Xr, yr)(Xg)
res = {"within_grounded_verify_oof": float(roc_auc_score(yg, oo)), "within_grounded_ci": boot_auc(yg, oo, gg), "ragtruth_within_oof": float(roc_auc_score(yr, oor)), "ragtruth_within_ci": boot_auc(yr, oor, gr),
       "grounded_to_ragtruth": float(roc_auc_score(yr, s)), "grounded_to_ragtruth_ci": boot_auc(yr, s, gr), "ragtruth_to_grounded": float(roc_auc_score(yg, s2)), "ragtruth_to_grounded_ci": boot_auc(yg, s2, gg), "n_ragtruth": int(len(yr)), "n_grounded": int(len(yg))}
print(json.dumps(res), flush=True)
t = json.load(open(f"{D}/transfer_cis.json")); t["grounding"] = res; t.pop("grounding_error", None); json.dump(t, open(f"{D}/transfer_cis.json", "w"), indent=1); print("updated transfer_cis.json")
