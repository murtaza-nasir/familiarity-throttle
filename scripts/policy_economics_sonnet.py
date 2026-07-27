#!/usr/bin/env python3
"""POLICY-LEVEL expected-cost economics of in-place correction (CAST) vs escalation (block-gate).

Distinct from the concurrent detection-threshold-vs-cost-ratio analysis (cost_model.json):
this is a POLICY comparison on the JUDGE-SCORED LIVE-AGENT loop (agentic_loop_cast_sonnet.json),
and it explicitly prices HUMAN ESCALATION (c_esc), which the threshold analysis omits.

Per-query expected cost (c_miss = 1 numeraire = cost of a lost answerable answer):

  E[cost] = pi * ( rho * P(acted-on fabrication | unsup) + c_esc * P(escalated | unsup) )
          + (1-pi) * ( c_esc * P(escalated | ans) + c_miss * (1 - utility_on_answerable) )

  rho    : cost of an acted-on (cascaded) fabrication relative to one lost answer
  c_esc  : cost of one human escalation relative to one lost answer
  pi     : prevalence of unsupported / knowledge-gap queries (eval is paired => pi=0.5)

Policies: no-gate (none), block-gate (escalate every flagged item), CAST (in-place correction),
abstention-prompt (prompt the model to refuse when unsure; utility craters).
"""
import json, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = os.environ.get("HN_DATA", "./data")
D = json.load(open(f"{P}/agentic_loop_cast_sonnet.json"))

# abstention-prompt behavioral numbers (instructed-to-abstain prompt), SONNET-JUDGED:
# instructed_arm_gens.jsonl -> instructed_arm_verdicts.jsonl (three-way judge, same
# instrument as every other arm). fictional COMMIT 28/378 = 0.074 (fab), answerable
# COMMIT 114/382 = 0.298 (utility). Refuses in place (no human escalation).
ABSTAIN = {"fab": 28/378, "fab_ci": [0.0498, 0.1053],
           "util": 114/382, "util_ci": [0.2530, 0.3471]}

TAU = "0.05"   # primary operating threshold (block escalates all unsupported, touches 0 answerable)
PI = 0.5       # paired eval => balanced unsupported/answerable

def wilson_lo_hi(node):
    """Return (rate, lo, hi) from a k/n node with ci95 if present."""
    r = node["rate"]
    ci = node.get("ci95", [r, r])
    return r, ci[0], ci[1]

def policy_rates(arm, tau):
    """Extract per-policy rates for one arm at one tau from the live loop."""
    A = D[arm]
    n_unsup = A["none"]["cascading_error"]["n"]
    n_ans   = A["none"]["utility_commit_answerable"]["n"]
    out = {}
    keymap = {"none": "none", "block": f"block@{tau}", "cast": f"cast@{tau}"}
    for pol, key in keymap.items():
        node = A[key]
        fab, fab_lo, fab_hi = wilson_lo_hi(node["cascading_error"])
        util, util_lo, util_hi = wilson_lo_hi(node["utility_commit_answerable"])
        # denominators: the policy's OWN arm counts (the none arm has one fewer
        # scoreable item, which previously produced an escalation rate of 73/72 > 1)
        esc_u = node.get("escalated_unsup", 0) / node["cascading_error"]["n"]
        esc_a = node.get("escalated_ans", 0) / node["utility_commit_answerable"]["n"]
        out[pol] = dict(fab=fab, fab_ci=[fab_lo, fab_hi],
                        util=util, util_ci=[util_lo, util_hi],
                        esc_unsup=esc_u, esc_ans=esc_a,
                        n_unsup=n_unsup, n_ans=n_ans,
                        k_fab=node["cascading_error"]["k"],
                        k_util=node["utility_commit_answerable"]["k"],
                        escalated_unsup=node.get("escalated_unsup", 0),
                        escalated_ans=node.get("escalated_ans", 0))
    # abstention-prompt (behavioral, prompt-only). Applied to the same arm's counts.
    out["abstain"] = dict(fab=ABSTAIN["fab"], fab_ci=ABSTAIN["fab_ci"],
                          util=ABSTAIN["util"], util_ci=ABSTAIN["util_ci"],
                          esc_unsup=0.0, esc_ans=0.0, n_unsup=n_unsup, n_ans=n_ans,
                          source="instructed_arm_verdicts.jsonl (Sonnet three-way judge)")
    # no-automation outside option: the step is not delegated; answerable queries go
    # unanswered (cost c_miss each), unanswerable queries produce no fabrication.
    out["null"] = dict(fab=0.0, fab_ci=None, util=0.0, util_ci=None,
                       esc_unsup=0.0, esc_ans=0.0, n_unsup=n_unsup, n_ans=n_ans,
                       source="definitional (no deployment)")
    return out

def ecost(r, rho, c_esc, pi=PI, c_miss=1.0):
    """Expected per-query cost for a policy-rate dict r."""
    return (pi * (rho * r["fab"] + c_esc * r["esc_unsup"])
            + (1 - pi) * (c_esc * r["esc_ans"] + c_miss * (1 - r["util"])))

def linear_coeffs(r, pi=PI):
    """E = a + b*rho + c*c_esc  (linear in the two swept axes)."""
    a = (1 - pi) * (1 - r["util"])
    b = pi * r["fab"]
    c = pi * r["esc_unsup"] + (1 - pi) * r["esc_ans"]
    return dict(const=a, rho=b, c_esc=c)

# ---- build for both arms ----
RESULT = {
    "_model": {
        "equation": "E = pi*(rho*P(fab|unsup)+c_esc*P(esc|unsup)) + (1-pi)*(c_esc*P(esc|ans)+c_miss*(1-util_ans))",
        "c_miss": 1.0, "pi": PI, "pi_rationale": "paired eval (1 real + 1 fictional per pair) => balanced",
        "tau_primary": TAU,
        "policies": ["none", "block", "cast", "abstain", "null"],
        "abstain_source": "instructed-arm generations judged by the same Sonnet three-way judge (instructed_arm_verdicts.jsonl); no human escalation",
        "distinct_from": "cost_model.json threshold-vs-cost-ratio analysis (this one prices escalation c_esc)",
    },
    "arms": {},
}

POLICY_COLORS = {"none": "#999999", "block": "#0072B2", "cast": "#E69F00", "abstain": "#CC79A7", "null": "#5c5c5c"}
POLICY_LABEL = {"none": "no-gate", "block": "block-gate (escalate)", "cast": "CAST (in-place)", "abstain": "abstention-prompt", "null": "no automation"}

# rho / c_esc grids
rho_grid = np.logspace(0, 3, 400)          # [1, 1000] log
cesc_grid = np.logspace(-1, np.log10(50), 400)   # [0.1, 50] log

def winner_grid(rates, pi=PI, policies=("none", "block", "cast", "abstain", "null")):
    W = np.empty((len(cesc_grid), len(rho_grid)), dtype=int)
    for i, ce in enumerate(cesc_grid):
        for j, rho in enumerate(rho_grid):
            costs = [ecost(rates[p], rho, ce, pi) for p in policies]
            W[i, j] = int(np.argmin(costs))
    return W

for arm in ["param", "grounded"]:
    rates = policy_rates(arm, TAU)
    coeffs = {p: linear_coeffs(rates[p]) for p in rates}
    # dominance: is CAST <= no-gate everywhere? (same util & esc, lower fab)
    cast_dominates_none = (rates["cast"]["fab"] <= rates["none"]["fab"]
                           and rates["cast"]["util"] >= rates["none"]["util"]
                           and rates["cast"]["esc_unsup"] <= rates["none"]["esc_unsup"]
                           and rates["cast"]["esc_ans"] <= rates["none"]["esc_ans"])
    # block-vs-cast boundary: block cheaper when c_esc*(escalation weight) < rho*(fab gap weight)
    # 0 = E_block - E_cast = [b_block-b_cast]*rho + [c_block-c_cast]*c_esc + [a_block-a_cast]
    bb, cb, ab = coeffs["block"]["rho"], coeffs["block"]["c_esc"], coeffs["block"]["const"]
    bc, cc, ac = coeffs["cast"]["rho"], coeffs["cast"]["c_esc"], coeffs["cast"]["const"]
    # E_block<E_cast  <=>  (cb-cc)*c_esc < (bc-bb)*rho + (ac-ab)
    # boundary slope of c_esc in rho: c_esc = [(bc-bb)*rho + (ac-ab)] / (cb-cc)
    boundary = {"c_esc_eq": f"c_esc = ({bc-bb:.5f}*rho + {ac-ab:.5f}) / {cb-cc:.5f}",
                "slope_cesc_per_rho": float((bc - bb) / (cb - cc)) if (cb - cc) else None,
                "intercept": float((ac - ab) / (cb - cc)) if (cb - cc) else None,
                "meaning": "below the line escalation is cheap => block wins; above it => CAST wins"}
    RESULT["arms"][arm] = {
        "n_unsup": rates["none"]["n_unsup"], "n_ans": rates["none"]["n_ans"],
        "rates": {p: {k: v for k, v in rates[p].items()} for p in rates},
        "linear_coeffs": coeffs,
        "cast_dominates_no_gate": bool(cast_dominates_none),
        "abstain_dominated_note": "utility crater (%.3f) makes abstention-prompt dominated whenever miss cost matters" % rates["abstain"]["util"],
        "block_vs_cast_boundary": boundary,
    }

# ---- CI sensitivity on the param arm block-vs-cast boundary ----
rp = policy_rates("param", TAU)
def boundary_slope(cast_fab, block_escu, block_esca, cast_escu, cast_esca, pi=PI):
    bc = pi * cast_fab; bb = pi * 0.0
    cc = pi * cast_escu + (1 - pi) * cast_esca
    cb = pi * block_escu + (1 - pi) * block_esca
    return (bc - bb) / (cb - cc)
base = boundary_slope(rp["cast"]["fab"], rp["block"]["esc_unsup"], rp["block"]["esc_ans"],
                      rp["cast"]["esc_unsup"], rp["cast"]["esc_ans"])
lo = boundary_slope(rp["cast"]["fab_ci"][0], rp["block"]["esc_unsup"], rp["block"]["esc_ans"],
                    rp["cast"]["esc_unsup"], rp["cast"]["esc_ans"])
hi = boundary_slope(rp["cast"]["fab_ci"][1], rp["block"]["esc_unsup"], rp["block"]["esc_ans"],
                    rp["cast"]["esc_unsup"], rp["cast"]["esc_ans"])
RESULT["ci_sensitivity_param"] = {
    "cast_fab_rate": rp["cast"]["fab"], "cast_fab_ci95": rp["cast"]["fab_ci"],
    "boundary_slope_cesc_per_rho": {"point": base, "at_fab_lo": lo, "at_fab_hi": hi},
    "interpretation": ("block-vs-CAST crossover is c_esc ~= slope*rho; at the CI edges of CAST's "
                       "residual fabrication rate the slope stays in [%.4f, %.4f], so the region map "
                       "ordering (block cheap-esc / CAST expensive-esc) is stable." % (lo, hi)),
}

# ---- concrete operating points (param arm) ----
ops = []
for (rho, ce, label) in [(5, 5, "moderate fab cost, mid escalation"),
                          (100, 0.5, "high fab cost, cheap escalation"),
                          (300, 40, "extreme fab cost, expensive escalation")]:
    costs = {p: ecost(rp[p], rho, ce) for p in ["none", "block", "cast", "abstain", "null"]}
    win = min(costs, key=costs.get)
    ops.append({"rho": rho, "c_esc": ce, "desc": label,
                "costs": {p: round(costs[p], 4) for p in costs},
                "winner": win, "winner_label": POLICY_LABEL[win]})
RESULT["operating_points_param"] = ops

json.dump(RESULT, open(f"{P}/policy_economics_sonnet.json", "w"), indent=2)
print("wrote policy_economics_sonnet.json")
print("param cast_dominates_no_gate:", RESULT["arms"]["param"]["cast_dominates_no_gate"])
print("param block-vs-cast boundary:", RESULT["arms"]["param"]["block_vs_cast_boundary"]["c_esc_eq"])
for o in ops:
    print(f"  rho={o['rho']:>4} c_esc={o['c_esc']:>4} -> {o['winner_label']:24s} costs={o['costs']}")

# ---------------- FIGURE ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
RHO, CE = np.meshgrid(rho_grid, cesc_grid)

for ax, arm, title in [(axes[0], "param", "(a) Parametric (knowledge-gap) arm"),
                        (axes[1], "grounded", "(b) Grounded (context) arm")]:
    rates = policy_rates(arm, TAU)
    policies = ["none", "block", "cast", "abstain", "null"]
    W = winner_grid(rates, policies=tuple(policies))
    present = sorted(np.unique(W))
    # build a colormap only for present winners
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([POLICY_COLORS[policies[k]] for k in present])
    remap = {k: i for i, k in enumerate(present)}
    Wr = np.vectorize(remap.get)(W)
    norm = BoundaryNorm(np.arange(-0.5, len(present) + 0.5), cmap.N)
    ax.pcolormesh(RHO, CE, Wr, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$  (cost of acted-on fabrication / lost answer)")
    ax.set_ylabel(r"$c_{\mathrm{esc}}$  (human-review cost / lost answer)")
    ax.set_title(title, fontsize=10)
    # operating points on panel (a)
    if arm == "param":
        for o in ops:
            ax.scatter([o["rho"]], [o["c_esc"]], c="k", s=26, zorder=5,
                       edgecolors="white", linewidths=0.8)
    legend_h = [Line2D([0], [0], marker="s", linestyle="", markersize=9,
                       markerfacecolor=POLICY_COLORS[policies[k]], markeredgecolor="none",
                       label=POLICY_LABEL[policies[k]]) for k in present]
    ax.legend(handles=legend_h, loc="upper left", fontsize=7.5, framealpha=0.92)

# fig.suptitle("Cost-minimizing policy over fabrication and escalation costs",
#              fontsize=10)
fig.savefig(f"{P}/policy_regions_sonnet.pdf", dpi=200)
print("wrote policy_regions_sonnet.pdf")
