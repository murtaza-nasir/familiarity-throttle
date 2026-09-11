#!/usr/bin/env python3
"""fig_familiarity with intervals: left = per-item probe scores (jitter), median bar, and 95% bootstrap interval on the entity-level mean;
right = commit and flag rates with Clopper-Pearson 95% intervals. Data: tmlr_revision/ladder_items.csv (from audit_section4.py)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv("tmlr_revision/ladder_items.csv")
ORDER = ["S1","S2","S3","S4","S5"]; LAB = ["fictional","strictly\nunseen","announced\nonly","low-exposure\nreleased","well-known"]
def cp(k,n):
    lo = 0 if k==0 else stats.beta.ppf(0.025,k,n-k+1); hi = 1 if k==n else stats.beta.ppf(0.975,k+1,n-k); return lo,hi
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8.6,3.3)); rng=np.random.default_rng(7)
for k,s in enumerate(ORDER):
    sub=df[df.stratum==s]; ps=sub.probe.values
    ax1.scatter(k+rng.uniform(-0.17,0.17,len(ps)), ps, s=9, alpha=0.35, color="#4878a8", linewidths=0)
    med=np.median(ps); ax1.plot([k-0.27,k+0.27],[med,med], color="#1a3a5c", lw=2.2, zorder=5)
    em=sub.groupby("entity").probe.mean().values; bs=rng.choice(em,(5000,len(em))).mean(1); lo,hi=np.percentile(bs,[2.5,97.5]); m=em.mean()
    ax1.errorbar(k+0.36, m, yerr=[[m-lo],[hi-m]], fmt="D", color="#b3543e", ms=4, capsize=3, lw=1.2, zorder=6)
ax1.set_xticks(range(5), LAB, fontsize=7.5); ax1.set_ylabel("probe score (fictional-likeness)", fontsize=8.5); ax1.set_ylim(-0.04,1.04)
ax1.set_title("Probe score by stratum (bar: median; diamond: entity mean, 95% CI)", fontsize=8.5); ax1.tick_params(axis="y", labelsize=8)
w=0.38; xs=np.arange(5)
for j,(cat,col,ec,lab) in enumerate([("COMMIT","#c9c9c9","#888","commit"),("FLAG","#b3543e","#7d3826","clean flag")]):
    vals=[]; errs=[[],[]]
    for s in ORDER:
        v=df[df.stratum==s].son; k_=int((v==cat).sum()); n=len(v); r=k_/n; lo,hi=cp(k_,n); vals.append(r); errs[0].append(r-lo); errs[1].append(hi-r)
    off = -w/2 if j==0 else w/2
    bars=ax2.bar(xs+off, vals, w, color=col, edgecolor=ec, lw=0.5, label=lab, yerr=errs, capsize=2.5, error_kw=dict(lw=0.9, ecolor="#222"))
    if cat=="COMMIT":
        for k in (0,1): bars[k].set_hatch("////"); bars[k].set_facecolor("#8a8a8a"); bars[k].set_edgecolor("#555")
ax2.set_xticks(range(5), LAB, fontsize=7.5); ax2.set_ylim(0,1.0); ax2.set_ylabel("rate (three-way judge)", fontsize=8.5)
ax2.set_title("Behavior by stratum (hatched: fabrication-certain commits)", fontsize=8.5); ax2.legend(fontsize=7.5, frameon=False, loc="upper left"); ax2.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig("figs/fig_familiarity.pdf"); print("wrote figs/fig_familiarity.pdf")
