#!/usr/bin/env python3
"""fig_familiarity: five-stratum exposure ladder (probe scores + behavior).

Inputs (replication_package/data/):
  graded_tier_entities.json           S2 recheck verdicts (26 clean of 40)
  graded_tier_gens.jsonl              original-run probe scores (S2/S3/S4 used)
  graded_tier_verdicts_sonnet.jsonl   Sonnet verdicts for the original run
  graded_tier_s1s5_restyled_gens.jsonl / _verdicts.jsonl  restyled S1/S5
Output: msom/papers/figs/fig_familiarity.pdf
"""
import json, os, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "..", "figs", "fig_familiarity.pdf")

s2 = json.load(open(f"{DATA}/graded_tier_entities.json"))["entities"]["S2"]
excluded = {e["entity"] for e in s2 if e.get("recheck_verdict") == "excluded"}

gens = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/graded_tier_gens.jsonl")}
verd = {}
for l in open(f"{DATA}/graded_tier_verdicts_sonnet.jsonl"):
    r = json.loads(l)
    if r["verdict"] in ("COMMIT", "DEFLECT", "FLAG"):
        verd[r["id"]] = r["verdict"]

rows = []
for i, g in gens.items():
    if g["stratum"] not in ("S2", "S3", "S4"):
        continue
    if g["stratum"] == "S2" and g["entity"] in excluded:
        continue
    if i in verd:
        rows.append((g["stratum"], g["probe_score"], verd[i]))

rg = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/graded_tier_s1s5_restyled_gens.jsonl")}
for l in open(f"{DATA}/graded_tier_s1s5_restyled_verdicts.jsonl"):
    r = json.loads(l)
    g = rg[r["id"]]
    rows.append((g["stratum"], g["probe_score"], r["verdict"]))

bg = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/graded_tier_s2b_gens.jsonl")}
for l in open(f"{DATA}/graded_tier_s2b_verdicts.jsonl"):
    r = json.loads(l)
    if r["verdict"] in ("COMMIT", "DEFLECT", "FLAG"):
        g = bg[r["id"]]
        rows.append(("S2", g["probe_score"], r["verdict"]))

ORDER = ["S1", "S2", "S3", "S4", "S5"]
LAB = ["fictional", "strictly\nunseen", "announced\nonly", "low-exposure\nreleased", "well-known"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
rng = np.random.default_rng(7)
for k, s in enumerate(ORDER):
    ps = [r[1] for r in rows if r[0] == s]
    x = k + rng.uniform(-0.16, 0.16, len(ps))
    ax1.scatter(x, ps, s=9, alpha=0.35, color="#4878a8", linewidths=0)
    med = statistics.median(ps)
    ax1.plot([k - 0.26, k + 0.26], [med, med], color="#1a3a5c", lw=2.2, zorder=5)
ax1.set_xticks(range(5), LAB, fontsize=7.5)
ax1.set_ylabel("probe score (fictional-likeness)", fontsize=8.5)
ax1.set_ylim(-0.04, 1.04)
ax1.set_title("Probe score by stratum (medians)", fontsize=9)
ax1.tick_params(axis="y", labelsize=8)

w = 0.38
commits = [np.mean([r[2] == "COMMIT" for r in rows if r[0] == s]) for s in ORDER]
flags = [np.mean([r[2] == "FLAG" for r in rows if r[0] == s]) for s in ORDER]
xs = np.arange(5)
ax2.bar(xs - w / 2, commits, w, color="#c9c9c9", edgecolor="#888", lw=0.5, label="commit")
ax2.bar(xs + w / 2, flags, w, color="#b3543e", edgecolor="#7d3826", lw=0.5, label="clean flag")
for k in (0, 1):  # fabrication-certain strata
    ax2.bar(xs[k] - w / 2, commits[k], w, color="#8a8a8a", edgecolor="#555", lw=0.7,
            hatch="////", zorder=3)
    ax2.text(xs[k] - w / 2, commits[k] + 0.02, f"{commits[k]:.2f}", ha="center", fontsize=7)
ax2.set_xticks(range(5), LAB, fontsize=7.5)
ax2.set_ylabel("rate", fontsize=8.5)
ax2.set_ylim(0, 1.04)
ax2.set_title("Flags decline along the ladder", fontsize=9)
ax2.tick_params(axis="y", labelsize=8)
from matplotlib.patches import Patch
handles = [Patch(facecolor="#c9c9c9", edgecolor="#888", lw=0.5, label="commit"),
           Patch(facecolor="#8a8a8a", edgecolor="#555", lw=0.7, hatch="////",
                 label="commit (every one\na fabrication)"),
           Patch(facecolor="#b3543e", edgecolor="#7d3826", lw=0.5, label="clean flag")]
ax2.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")

ax1.annotate("never existed;\nheavily covered", xy=(2.05, 0.03), xytext=(2.35, 0.30),
             fontsize=7, color="#1a3a5c", ha="left",
             arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=0.8))
ax1.annotate("long released;\nsparsely covered", xy=(3.28, 0.90), xytext=(3.42, 0.62),
             fontsize=7, color="#1a3a5c", ha="left",
             arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=0.8))

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
print("commits", [round(c, 3) for c in commits])
print("flags  ", [round(f, 3) for f in flags])
