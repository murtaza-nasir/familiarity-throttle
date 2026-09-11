import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
t=json.load(open("tmlr_revision/transfer_cis.json")); q=t["qwen3_8b"]; g=t.get("grounding",{})
# PopQA tertile AUROC with bootstrap CI over items
import pandas as pd
pq=pd.read_json("tmlr_revision/popqa_external.jsonl", lines=True); lp=pq.log_pop.values; ps=pq.probe_score.values
qs=np.quantile(lp,[1/3,2/3]); lo=lp<=qs[0]; hi=lp>=qs[1]; y=np.r_[np.ones(lo.sum()),np.zeros(hi.sum())]; s=np.r_[ps[lo],ps[hi]]
rng=np.random.default_rng(0); bs=[]
for _ in range(2000):
    i=rng.integers(0,len(y),len(y)); bs.append(roc_auc_score(y[i],s[i]))
pop=(roc_auc_score(y,s), np.percentile(bs,2.5), np.percentile(bs,97.5))
bars=[("familiarity\nwithin domain", q["within_scm_oof"], q["within_scm_ci"], "#2a5d8f"),
      ("familiarity\ncross-domain", q["cross_all_from_scm"], q["cross_all_ci"], "#2a5d8f"),
      ("familiarity\nPopQA (external)", pop[0], [pop[1],pop[2]], "#2a5d8f"),
      ("grounding\nwithin corpus", g.get("ragtruth_within_oof", 0.717), g.get("ragtruth_within_ci",[0.717,0.717]), "#b3543e"),
      ("grounding\ncross-corpus", g.get("grounded_to_ragtruth", 0.599), g.get("grounded_to_ragtruth_ci",[0.599,0.599]), "#b3543e")]
fig,ax=plt.subplots(figsize=(5.8,3.0))
for i,(lab,v,ci,col) in enumerate(bars):
    ax.bar(i,v,color=col,width=0.62,edgecolor="#333",lw=0.5); ax.errorbar(i,v,yerr=[[v-ci[0]],[ci[1]-v]],fmt="none",ecolor="#111",capsize=3,lw=1)
    ax.text(i,v+0.015,f"{v:.2f}",ha="center",fontsize=8)
ax.axhline(0.5,color="#999",lw=0.7,ls=":"); ax.set_ylim(0.4,1.05); ax.set_xticks(range(5),[b[0] for b in bars],fontsize=7.5); ax.set_ylabel("AUROC",fontsize=9); ax.tick_params(axis="y",labelsize=8)
fig.tight_layout(); fig.savefig("figs/fig_generalization.pdf"); print("wrote fig_generalization.pdf", pop)
