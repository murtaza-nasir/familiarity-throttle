import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("replication_package/results/mechanism/throttle_trace_v2.json")); L=sorted(int(k) for k in d["by_layer"])
au=[d["decodability_auroc"][str(l)] for l in L]; eff=[d["by_layer"][str(l)]["delta_above_placebo"] for l in L]
causal=[d["by_layer"][str(l)]["causal_logitdiff"] for l in L]; plac=[d["by_layer"][str(l)]["placebo_logitdiff"] for l in L]
lo=[d["by_layer"][str(l)]["ci_low"]-d["by_layer"][str(l)]["placebo_logitdiff"] for l in L]; hi=[d["by_layer"][str(l)]["ci_high"]-d["by_layer"][str(l)]["placebo_logitdiff"] for l in L]
fig,ax=plt.subplots(figsize=(5.6,3.2)); ax2=ax.twinx()
ax.plot(L,au,"o-",color="#2a5d8f",ms=3,lw=1.5,label="decodability (probe AUROC)"); ax.set_ylabel("AUROC",color="#2a5d8f",fontsize=9); ax.set_ylim(0.9,1.005)
ax2.fill_between(L,lo,hi,color="#d9822b",alpha=0.22,lw=0); ax2.plot(L,eff,"s-",color="#d9822b",ms=3,lw=1.5,label="causal effect above placebo")
ax2.axhline(0,color="#999",lw=0.7,ls=":"); ax2.set_ylabel("abstain$-$commit logit shift (above norm-matched placebo)",color="#d9822b",fontsize=8)
ax.set_xlabel("residual-stream position (0 = embeddings; $\\ell$ = after block $\\ell$)",fontsize=9); ax.axvspan(15,22,color="#d9822b",alpha=0.08,lw=0)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,fontsize=7.5,frameon=False,loc="lower right")
ax.tick_params(labelsize=8); ax2.tick_params(labelsize=8); fig.tight_layout(); fig.savefig("figs/fig_throttle.pdf"); print("wrote fig_throttle.pdf")
