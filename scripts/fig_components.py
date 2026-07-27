#!/usr/bin/env python3
"""fig_components: readout concentration + top-component effects (Qwen3-8B).

Input: results/mechanism/component_throttle.json
Output: msom/papers/figs/fig_components.pdf
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "..", "results", "mechanism", "component_throttle.json")))
OUT = os.path.join(HERE, "..", "..", "figs", "fig_components.pdf")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.1))

conc = d["concentration_true_patching"]
ks = [c["k"] for c in conc]
fr = [100 * c["frac_of_full"] for c in conc]
ax1.plot(ks, fr, "o-", color="#2a5d8f", lw=2, ms=6)
for k, f in zip(ks, fr):
    ax1.annotate(f"{f:.0f}%", (k, f), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8.5)
ax1.set_xscale("log")
ax1.axhline(100, color="#999", lw=0.7, ls=":")
ax1.set_ylim(0, 112)
ax1.set_xlabel("top-$k$ components (of 1,188)", fontsize=9)
ax1.set_ylabel("% of familiarity gap closed", fontsize=9)
ax1.set_title("Concentration of the readout", fontsize=10)
ax1.tick_params(labelsize=8.5)

top = d["verified_effects"][:5]
names = [f"L{c['block']}\n{'MLP' if c['head'] is None else 'H%d' % c['head']}" for c in top]
vals = [abs(c["true_effect_mean"]) for c in top]
errs = [[abs(c["true_effect_mean"]) - abs(c["ci_high"]) for c in top],
        [abs(c["ci_low"]) - abs(c["true_effect_mean"]) for c in top]]
cols = ["#d9822b" if c["head"] is None else "#2a5d8f" for c in top]
ax2.bar(range(5), vals, yerr=errs, color=cols, capsize=3,
        error_kw=dict(lw=1, ecolor="#222"))
ax2.set_xticks(range(5), names, fontsize=8.5)
ax2.set_ylabel(r"|$\Delta$ logit-diff| (true patching)", fontsize=9)
ax2.set_title("Top components", fontsize=10)
ax2.legend(handles=[Patch(facecolor="#d9822b", label="MLP"),
                    Patch(facecolor="#2a5d8f", label="attention head")],
           fontsize=8, frameon=False, loc="upper right")
ax2.tick_params(labelsize=8.5)

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
print("concentration:", [f"{k}:{f:.0f}%" for k, f in zip(ks, fr)])
