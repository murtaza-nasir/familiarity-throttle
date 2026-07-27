#!/usr/bin/env python3
"""Independent verification pass for w2_numerics.py (deliverable 5).

Every headline number is recomputed by an independent route or checked against a
sanity identity:
  V1  Erlang-C: recursion vs direct factorial-sum formula (grid).
  V2  L_q(s,.) strictly increasing and convex in Lambda; W_q increasing (grid).
  V3  Pi*(rho) nondecreasing in rho (envelope of nondecreasing affine maps).
  V4  rho* closed forms: escalation-free frontier equals (1-pi)/(pi*fC) = 1/fC at
      pi=0.5 on the empirical Qwen axis (answer-below-support absorbs d_C), and
      rho* = inf for cesc < 1 with the measured (perfectly separated) gate.
  V5  Clopper-Pearson 0/73 upper bound: scipy beta.ppf vs closed form 1-(alpha/2)^(1/n).
  V6  Desk cells: joint optimum re-derived by a brute-force second implementation
      (direct triple loop over (i, j, s), midpoint thresholds) on a subsample.
  V7  Wedge cost >= 0 in every cell; QED relative gap >= 0 in every cell.
  V8  lambda-dagger: sign of J*-J0 checked just below / above each reported finite
      crossing with an independent J* evaluation.
  V9  MLR summary: AUROC of the param block gate = 1.0 (perfect separation) via
      Mann-Whitney; grounded violation mass recomputed from the stored bin counts.

Writes results/w2_verify.json; exits nonzero on any failure.
"""
import json, math, os, sys
import numpy as np
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RES = os.path.join(PKG, "results")
sys.path.insert(0, HERE)
import w2_numerics as W

checks = {}
fail = []

def record(name, ok, detail):
    checks[name] = dict(ok=bool(ok), detail=detail)
    if not ok:
        fail.append(name)

# ---- V1 Erlang-C independent formula
def erlang_c_direct(s, a):
    if a >= s: return 1.0
    from math import factorial
    top = a**s/factorial(s) * s/(s-a)
    bot = sum(a**k/factorial(k) for k in range(s)) + top
    return top/bot

errs = []
for s in [1, 2, 5, 12, 40, 80]:
    for util in [0.05, 0.3, 0.6, 0.9, 0.98]:
        a = util*s
        errs.append(abs(W.erlang_c(s, a) - erlang_c_direct(s, a)))
record("V1_erlangC_two_impls", max(errs) < 1e-9, dict(max_abs_err=float(max(errs))))

# ---- V2 monotonicity / convexity of Lq, monotone Wq
ok = True; worst = None
for s in [1, 3, 8, 40]:
    mu = 12.0
    lam = np.linspace(0.05*s*mu, 0.995*s*mu, 250)
    L = np.array([W.Lq(s, x, mu) for x in lam])
    Wv = np.array([W.Wq(s, x, mu) for x in lam])
    dL = np.diff(L); d2L = np.diff(dL); dW = np.diff(Wv)
    if not (np.all(dL > 0) and np.all(d2L > -1e-9) and np.all(dW > 0)):
        ok = False; worst = s
record("V2_Lq_monotone_convex_Wq_monotone", ok, dict(failed_s=worst))

# ---- load main results and rebuild distributions
out = json.load(open(os.path.join(RES, "w2_numerics.json")))
scores = W.load_scores()
u_q, a_q = scores[("param", "block_gate_score")]
qwen = W.Dist.from_samples(u_q, a_q)

# ---- V3 Pi*(rho) nondecreasing
curve = out["static_frontier"]["_pi_star_curve_qwen_cesc0.05_dC0"]
vals = [c[1] for c in curve]
record("V3_pi_star_nondecreasing", all(vals[i+1] >= vals[i]-1e-12 for i in range(len(vals)-1)),
       dict(n_points=len(vals)))

# ---- V4 rho* closed forms
fC = 6/73
r_escfree = W.rho_star(qwen, math.inf, 60/72, fC, 0.0)
r_escfree_dub = W.rho_star(qwen, math.inf, 60/72, fC, W.DC_UB)
r_small = [W.rho_star(qwen, c0, 60/72, fC, 0.0) for c0 in (0.01, 0.05, 0.1)]
record("V4_rho_star_closed_forms",
       abs(r_escfree - 1/fC) < 1e-6 and abs(r_escfree_dub - 1/fC) < 1e-6
       and all(math.isinf(x) for x in r_small),
       dict(escalation_free=r_escfree, escalation_free_dC_ub=r_escfree_dub,
            one_over_fC=1/fC, small_c0=[str(x) for x in r_small],
            note=("empirical axis: answer region below the unanswerable support "
                  "absorbs the d_C term, so the escalation-free frontier is 1/fC "
                  "for BOTH d_C values; the two-atom CP-gate variant moves it to "
                  "(1-dC)/fC as reported in w2_numerics.json")))

# ---- V5 CP bound closed form
cf = 1 - (0.025)**(1/73)
record("V5_cp_closed_form", abs(cf - W.DC_UB) < 1e-12,
       dict(closed_form=cf, scipy=W.DC_UB))

# ---- V6 brute-force desk re-optimization on a subsample
def desk_brute(dist, lam, mu, h, cs, rho, kappa, fC, dC, pi=0.5):
    best = math.inf
    n = len(dist.T)
    for j in range(n):
        lam_esc = lam*(pi*dist.Eu[j] + (1-pi)*dist.Ea[j])
        for i in range(j+1):
            disp = (pi*(kappa*rho*dist.Gu[i] + fC*rho*(dist.Gu[j]-dist.Gu[i]))
                    + (1-pi)*dC*(dist.Ga[j]-dist.Ga[i]))
            if lam_esc <= 0:
                best = min(best, lam*disp)
                continue
            smax = int(math.ceil(lam_esc/mu)) + 60
            for s in range(int(math.floor(lam_esc/mu))+1, smax):
                best = min(best, lam*disp + h*W.Lq(s, lam_esc, mu) + cs*s)
    return best

rng = np.random.default_rng(0)
cells = out["desk"]["cells"]
idx = rng.choice(len(cells), size=8, replace=False)
worst_gap = 0.0
for k in idx:
    c = cells[int(k)]
    jb = desk_brute(qwen, c["lam"], 12.0, c["h"], c["cs"], c["rho"], 60/72, 6/73, 0.0)
    worst_gap = max(worst_gap, abs(jb - c["exact"]["J"])/max(1e-9, c["exact"]["J"]))
record("V6_desk_brute_force_match", worst_gap < 1e-9,
       dict(n_cells_checked=len(idx), worst_rel_diff=float(worst_gap)))

# ---- V7 wedge and QED gap signs
w_ok = all(c["wedge"]["cost"] >= -1e-9 for c in cells if "wedge" in c)
q_ok = all(c["qed"]["rel_gap"] >= -1e-9 for c in cells)
wg = out["wedge_grounded"]["cells"]
wg_ok = all(c["wedge"]["cost"] >= -1e-9 for c in wg if "wedge" in c)
over_esc_neg = [c for c in wg if "wedge" in c and c["wedge"]["over_escalation"] < -1e-9]
record("V7_wedge_qed_signs", w_ok and q_ok and wg_ok,
       dict(param_wedges_nonneg=w_ok, qed_gaps_nonneg=q_ok, grounded_wedges_nonneg=wg_ok,
            grounded_under_escalation_cells=len(over_esc_neg),
            note=("under-escalation cells on the grounded ROC are expected: the "
                  "grounded ROC violates MLR, so Cor. 1's sign is not guaranteed there")))

# ---- V8 lambda-dagger crossings
ld_ok = True; ld_detail = {}
for key, v in out["lambda_dagger"]["results"].items():
    ld = v.get("lam_dagger")
    if not isinstance(ld, float):
        continue
    # independent J* evaluation at fixed s
    import re
    m = re.match(r"rho=([\d.]+)\|dC=([\d.]+)\|(\w+)\|cs=([\d.]+)\|s=(\d+)", key)
    rho, dC, gtag, cs, s = float(m.group(1)), float(m.group(2)), m.group(3), float(m.group(4)), int(m.group(5))
    dist = qwen if gtag == "empirical" else W.Dist.two_atom(W.EU_LB, W.EA_UB)
    h = 0.5*rho
    def Jstar(lam):
        best = math.inf
        for j in range(len(dist.T)):
            lam_esc = lam*(0.5*dist.Eu[j] + 0.5*dist.Ea[j])
            A = 0.5*(60/72 - 6/73)*rho*dist.Gu - 0.5*dC*dist.Ga
            i = int(np.argmin(A[:j+1]))
            disp = (0.5*((60/72)*rho*dist.Gu[i] + (6/73)*rho*(dist.Gu[j]-dist.Gu[i]))
                    + 0.5*dC*(dist.Ga[j]-dist.Ga[i]))
            cong = h*W.Lq(s, lam_esc, 12.0) if lam_esc < s*12.0 else math.inf
            best = min(best, lam*disp + cong)
        return best + cs*s
    lo, hi = 0.98*ld, 1.02*ld
    ok = (Jstar(lo) < 0.5*lo) and (Jstar(hi) >= 0.5*hi)
    if not ok:
        ld_ok = False; ld_detail[key] = dict(ld=ld, J_lo=Jstar(lo), J0_lo=0.5*lo,
                                             J_hi=Jstar(hi), J0_hi=0.5*hi)
record("V8_lambda_dagger_crossings", ld_ok, ld_detail or dict(all="verified"))

# ---- V9 MLR summaries
from scipy.stats import mannwhitneyu
aur = mannwhitneyu(u_q, a_q, alternative="two-sided").statistic/(len(u_q)*len(a_q))
m = out["mlr"]["grounded|block_gate_score"]
mass = sum((m["count_u"][v["bin"]]+m["count_a"][v["bin"]])
           for v in m["violations"])/(sum(m["count_u"])+sum(m["count_a"]))
record("V9_mlr_summaries",
       abs(aur - 1.0) < 1e-12 and abs(mass - m["violation_traffic_mass"]) < 1e-9,
       dict(param_block_auroc=float(aur), grounded_violation_mass_recomputed=float(mass)))

with open(os.path.join(RES, "w2_verify.json"), "w") as f:
    json.dump(dict(all_passed=len(fail) == 0, failed=fail, checks=checks), f, indent=1)
print("wrote results/w2_verify.json;", "ALL PASSED" if not fail else f"FAILED: {fail}")
sys.exit(0 if not fail else 1)
