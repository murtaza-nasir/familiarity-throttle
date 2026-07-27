import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
P=os.environ.get("HN_DATA", "./data")
L=18
zt=np.load(P+"/repr/temporal_qwen3.npz",allow_pickle=True)
tier=zt["tier"].astype(str); Xt=zt["meanpool"][L].astype(np.float32)
cen=json.load(open(P+"/postcutoff_verification_census.json"))
valid_idx={int(r["id"].rsplit("_",1)[1]) for r in cen if r["verdict"]=="VALID"}
pcpos=np.where(tier=="postcutoff")[0]
pc_valid=np.array([pcpos[i] for i in sorted(valid_idx)])
wk=np.where(tier=="wellknown")[0]
z=np.load(P+"/repr/qwen3_instruct.npz",allow_pickle=True)
pm=z["arm"]=="parametric"
Xe=z["meanpool"][L,pm,:].astype(np.float32); ye=z["y"][pm].astype(int)
# variant 1: wellknown vs VERIFIED postcutoff only
Xtr=np.vstack([Xt[wk],Xt[pc_valid]]); ytr=np.r_[np.zeros(len(wk)),np.ones(len(pc_valid))].astype(int)
sc=StandardScaler().fit(Xtr); clf=LogisticRegression(C=0.3,max_iter=2000).fit(sc.transform(Xtr),ytr)
print("label-free v2 (wellknown vs VERIFIED postcutoff, n=%d): main testbed AUROC=%.4f" % (len(ytr), roc_auc_score(ye,clf.decision_function(sc.transform(Xe)))))
# variant 2: perplexity pseudo-labels (no human/generated labels): top/bottom terciles of entity perplexity
ppl=json.load(open(P+"/entity_ppl.json"))
key=[k for k in ppl.keys() if "per_item" in k or "items" in k or "ppl" in k]
print("entity_ppl keys:", list(ppl.keys())[:8])
