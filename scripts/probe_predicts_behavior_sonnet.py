#!/usr/bin/env python3
"""#7 construct validity: probe predicts ACTUAL model behavior (3-way judge), not just designed label.
OOF probe = GroupKFold(5) grouped by pair id, StandardScaler+LogReg(C=0.3), meanpool L18 on
repr/qwen3_instruct.npz (arm=='parametric'). Behavior = 3-way judge verdict per item, aligned by
index (crossmodel_raw/qwen3.jsonl order == items parametric order == npz parametric order)."""
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

P = os.environ.get("HN_ROOT", "."); D = f"{P}/data"
items = [json.loads(l) for l in open(f"{D}/items.jsonl") if json.loads(l)["arm"] == "parametric"]
z = np.load(f"{D}/repr/qwen3_instruct.npz", allow_pickle=True)
mask = z["arm"] == "parametric"
X = z["meanpool"][18, mask, :].astype(np.float32)
y = z["y"][mask].astype(int)                       # 1 = should_abstain (fictional)
grp = np.array([it["id"].rsplit("_", 1)[0] for it in items])
assert len(items) == mask.sum() == len(X), (len(items), int(mask.sum()), len(X))
# sanity: y from npz matches label order
y_lab = np.array([1 if it["label"] == "should_abstain" else 0 for it in items])
assert (y == y_lab).all(), "label/order mismatch"

# OOF probe score (prob of fictional/abstain)
gkf = GroupKFold(5); oof = np.zeros(len(y))
for tr, te in gkf.split(X, y, grp):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=0.3, solver="liblinear", max_iter=1000).fit(sc.transform(X[tr]), y[tr])
    oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
print("OOF probe AUROC (designed label real-vs-fictional): %.4f" % roc_auc_score(y, oof))

# per-item behavior verdicts (same order)
verds = [json.loads(l)["verdict"] for l in open(f"{D}/sonnet_qwen3_peritem.jsonl")]
assert len(verds) == len(items), (len(verds), len(items))
verds = np.array(verds)

fic = (y == 1)
commit = (verds == "COMMIT")
flag = (verds == "FLAG")
deflect = (verds == "DEFLECT")


def group_boot_auroc(score, target, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    base = roc_auc_score(target, score)
    vals = []
    for _ in range(n):
        samp = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        tt = target[idx]
        if len(np.unique(tt)) < 2:
            continue
        vals.append(roc_auc_score(tt, score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, lo, hi, len(vals)


out = {}
# Framing 1 (task primary): among fictional, probe-fictional-score predicts FABRICATE(COMMIT) vs {FLAG,DEFLECT}
s1 = oof[fic]; t1 = commit[fic].astype(int); g1 = grp[fic]
a, lo, hi, nb = group_boot_auroc(s1, t1, g1)
out["fic_probe_predicts_COMMIT_vs_rest"] = {"auroc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
    "n_items": int(fic.sum()), "n_commit": int(t1.sum()), "n_boot": nb,
    "desc": "among fictional items, probe fictional score predicting COMMIT(fabricate)=1 vs FLAG/DEFLECT=0"}

# Framing 2: among fictional, COMMIT vs FLAG only (drop DEFLECT)
sel = fic & (commit | flag)
s2 = oof[sel]; t2 = commit[sel].astype(int); g2 = grp[sel]
a, lo, hi, nb = group_boot_auroc(s2, t2, g2)
out["fic_probe_predicts_COMMIT_vs_FLAG"] = {"auroc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
    "n_items": int(sel.sum()), "n_commit": int(t2.sum()), "n_flag": int((flag[sel]).sum()), "n_boot": nb,
    "desc": "among fictional COMMIT|FLAG items, probe fictional score predicting COMMIT vs FLAG"}

# Framing 3: across ALL parametric items, probe predicts behavioral hallucination = (fictional AND COMMIT)
halluc = (fic & commit).astype(int)
a, lo, hi, nb = group_boot_auroc(oof, halluc, grp)
out["all_probe_predicts_hallucination_event"] = {"auroc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
    "n_items": int(len(y)), "n_halluc": int(halluc.sum()), "n_boot": nb,
    "desc": "across all parametric items, probe fictional score predicting behavioral hallucination (fictional AND COMMIT)=1"}

# verdict rate breakdown by designed label (for context)
for lab, m in [("fictional", fic), ("real", ~fic)]:
    tot = int(m.sum())
    out[f"verdict_rates_{lab}"] = {v: round(float((verds[m] == v).mean()), 3) for v in ["COMMIT", "DEFLECT", "FLAG"]}
    out[f"verdict_rates_{lab}"]["n"] = tot

json.dump(out, open(f"{D}/probe_predicts_behavior_sonnet.json", "w"), indent=2)
print(json.dumps(out, indent=2))
