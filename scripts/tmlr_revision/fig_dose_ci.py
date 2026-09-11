import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
d=json.load(open("replication_package/results/mechanism/dose_response_summary.json"))
def cp(r,n):
    k=round(r*n); lo=0 if k==0 else stats.beta.ppf(0.025,k,n-k+1); hi=1 if k==n else stats.beta.ppf(0.975,k+1,n-k); return r-lo,hi-r
al=[0.0,0.1,0.25,0.4,0.56,0.75,1.0,1.25]; keys=[f"opt_x{a}" for a in al]
ff=[d[k]["flag_fic"] for k in keys]; fr=[d[k]["ff_real"] for k in keys]
eff=np.array([cp(d[k]["flag_fic"],d[k]["n_fic"]) for k in keys]).T; efr=np.array([cp(d[k]["ff_real"],d[k]["n_real"]) for k in keys]).T
fig,ax=plt.subplots(figsize=(5.4,3.2))
ax.errorbar(al,ff,yerr=eff,fmt="o-",color="#b3543e",capsize=2.5,lw=1.5,ms=4,label="flags on fictional entities")
ax.errorbar(al,fr,yerr=efr,fmt="s--",color="#2a5d8f",capsize=2.5,lw=1.5,ms=4,label="false-flags on real entities")
m=d["meandiff_natural"]; e=cp(m["flag_fic"],m["n_fic"]); ax.errorbar([0.56],[m["flag_fic"]],yerr=[[e[0]],[e[1]]],fmt="D",mfc="white",color="#b3543e",ms=6,capsize=2.5,label="mean-difference direction at its natural norm")
ax.set_xlabel("steering magnitude (fraction of trained norm; 0.56 = natural mean-difference norm)",fontsize=8.5); ax.set_ylabel("rate (three-way judge)",fontsize=9); ax.set_ylim(0,1.02)
ax.legend(fontsize=7.5,frameon=False,loc="upper left"); ax.tick_params(labelsize=8); fig.tight_layout(); fig.savefig("figs/fig_dose.pdf"); print("wrote fig_dose.pdf")
