#!/usr/bin/env python3
"""W2 numerics: static frontier verification, boundary/degeneracy treatment, MLR check,
worked M/M/s desk example, and the QED exact-vs-approximate gap table (EC.2.6).

Model: w2_derivations.md (two-threshold disposition on a probe-score axis + M/M/s desk).
Empirical score distributions: per-item BLOCK-gate probe scores from the live-agent
calibration (results/cast_stage_items.jsonl, param loop) -- the same items behind
Table 4 / policy_economics_sonnet.json. Qwen3-8B parametric arm: 73 answerable /
73 unsupported, judged rates kappa=60/72, f_C=6/73, d_C=0/73, gate E_u=73/73, E_a=0/73.
Llama-3.1-8B: no per-item gate scores ship in the replication package, so Llama is
represented by its operating-point gate rates as a two-atom score distribution
(E_u=69/73, E_a=2/73), which is exactly the information the paper's Llama numbers use.

All costs in the lost-answer numeraire; pi = 0.5 (balanced paired eval).

Outputs: results/w2_numerics.json
"""
import json, math, os
import numpy as np
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RES = os.path.join(PKG, "results")

PI = 0.5

# ---------------------------------------------------------------- primitives
def cp_interval(k, n, conf=0.95):
    """Clopper-Pearson exact two-sided interval."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi

# ---- measured rates (counts straight from the shipped result files / paper Sec. 6)
QWEN = dict(kappa=60/72, fC=6/73, dC=0/73,
            k_kappa=(60,72), k_fC=(6,73), k_dC=(0,73), k_Eu=(73,73), k_Ea=(0,73))
LLAMA = dict(kappa=72/73, fC=50/73, dC=0/73,
             k_kappa=(72,73), k_fC=(50,73), k_dC=(0,73), k_Eu=(69,73), k_Ea=(2,73))

DC_UB = cp_interval(0, 73)[1]          # 0.04928 (referee point M1)
EU_LB = cp_interval(73, 73)[0]         # 0.95072
EA_UB = cp_interval(0, 73)[1]          # 0.04928

# ---------------------------------------------------------------- score data
def load_scores():
    lines = [json.loads(l) for l in open(os.path.join(RES, "cast_stage_items.jsonl"))]
    out = {}
    for loop in ("param", "grounded"):
        for key in ("block_gate_score", "cast_gate_score"):
            u = np.array(sorted(l[key] for l in lines
                                if l["loop"] == loop and l["label"] == "should_abstain"))
            a = np.array(sorted(l[key] for l in lines
                                if l["loop"] == loop and l["label"] == "answerable"))
            out[(loop, key)] = (u, a)
    return out

# ---------------------------------------------------------------- empirical dist
class Dist:
    """Weighted empirical score distribution for the two classes.

    Threshold candidates T = unique pooled values plus +inf; the policy
    'escalate x >= tau' / 'answer x < tau_c' is evaluated on exact empirical mass
    (atoms respected -- no interpolation)."""
    def __init__(self, vals_u, pmf_u, vals_a, pmf_a):
        self.vals_u, self.pmf_u = np.asarray(vals_u, float), np.asarray(pmf_u, float)
        self.vals_a, self.pmf_a = np.asarray(vals_a, float), np.asarray(pmf_a, float)
        T = np.unique(np.concatenate([self.vals_u, self.vals_a]))
        self.T = np.concatenate([T, [np.inf]])
        cu = np.concatenate([[0.0], np.cumsum(self.pmf_u[np.argsort(self.vals_u)])])
        # tail masses via searchsorted on sorted values
        su, sa = np.sort(self.vals_u), np.sort(self.vals_a)
        wu = self.pmf_u[np.argsort(self.vals_u)]
        wa = self.pmf_a[np.argsort(self.vals_a)]
        cum_u = np.concatenate([[0.0], np.cumsum(wu)])
        cum_a = np.concatenate([[0.0], np.cumsum(wa)])
        iu = np.searchsorted(su, self.T, side="left")
        ia = np.searchsorted(sa, self.T, side="left")
        self.Gu = cum_u[iu] / cum_u[-1]          # P(X_u < tau)
        self.Ga = cum_a[ia] / cum_a[-1]
        self.Eu = 1.0 - self.Gu                  # P(X_u >= tau)
        self.Ea = 1.0 - self.Ga

    @classmethod
    def from_samples(cls, u, a):
        return cls(u, np.full(len(u), 1/len(u)), a, np.full(len(a), 1/len(a)))

    @classmethod
    def two_atom(cls, Eu, Ea):
        """Operating-point representation: score in {0,1}, P(1|u)=Eu, P(1|a)=Ea."""
        return cls([0.0, 1.0], [1-Eu, Eu], [0.0, 1.0], [1-Ea, Ea])

# ---------------------------------------------------------------- static objective
def static_best(dist, rho, cesc, kappa, fC, dC, pi=PI):
    """Min per-query cost over two-threshold policies (tau_c <= tau_b on candidate grid).

    Pi(i,j) = pi[kappa rho Gu_i + fC rho (Gu_j-Gu_i) + cesc Eu_j]
            + (1-pi)[dC (Ga_j-Ga_i) + cesc Ea_j]
    separates as A(i)+B(j); enforce i<=j by prefix-min of A."""
    A = pi*(kappa - fC)*rho*dist.Gu - (1-pi)*dC*dist.Ga
    if math.isinf(cesc):
        B = pi*fC*rho*dist.Gu + (1-pi)*dC*dist.Ga + np.where(dist.Eu + dist.Ea > 0, np.inf, 0.0)
        B[-1] = pi*fC*rho*dist.Gu[-1] + (1-pi)*dC*dist.Ga[-1]  # tau_b=+inf: no escalation
    else:
        B = pi*(fC*rho*dist.Gu + cesc*dist.Eu) + (1-pi)*(dC*dist.Ga + cesc*dist.Ea)
    prefA = np.minimum.accumulate(A)
    idxA = np.zeros(len(A), dtype=int)
    best = A[0]
    for i in range(len(A)):
        if A[i] < best:
            best = A[i]; idxA[i] = i
        else:
            idxA[i] = idxA[i-1] if i else 0
    tot = prefA + B
    j = int(np.argmin(tot))
    i = int(idxA[j])
    return dict(cost=float(tot[j]), i=i, j=j,
                tau_c=float(dist.T[i]), tau_b=float(dist.T[j]),
                Gu_c=float(dist.Gu[i]), Gu_b=float(dist.Gu[j]),
                Ea_b=float(dist.Ea[j]), Eu_b=float(dist.Eu[j]))

def rho_star(dist, cesc, kappa, fC, dC, pi=PI, rho_max=1e7):
    """rho* = sup{rho: Pi*(rho) < 1-pi} by bisection (Pi* nondecreasing in rho)."""
    f = lambda r: static_best(dist, r, cesc, kappa, fC, dC, pi)["cost"] - (1-pi)
    if f(rho_max) < 0:
        return math.inf
    lo, hi = 1e-6, rho_max
    if f(lo) >= 0:
        return 0.0
    for _ in range(200):
        mid = math.sqrt(lo*hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    return float(math.sqrt(lo*hi))

# ---------------------------------------------------------------- Erlang C / M/M/s
def erlang_c(s, a):
    """Delay probability, via Erlang-B recursion (stable)."""
    if a <= 0: return 0.0
    if a >= s: return 1.0
    B = 1.0
    for k in range(1, s+1):
        B = a*B/(k + a*B)
    return s*B/(s - a*(1-B))

def Lq(s, lam_esc, mu):
    a = lam_esc/mu
    if lam_esc <= 0: return 0.0
    if a >= s: return math.inf
    C = erlang_c(s, a)
    return C*a/(s - a)

def Wq(s, lam_esc, mu):
    if lam_esc <= 0: return 0.0
    if lam_esc >= s*mu: return math.inf
    return erlang_c(s, lam_esc/mu)/(s*mu - lam_esc)

# ---------------------------------------------------------------- joint desk optimum
def desk_exact(dist, lam, mu, h, cs, rho, kappa, fC, dC, pi=PI, s_extra=30):
    """Exact joint optimum of J = lam*dispatch + h*Lq(s,Lam_esc) + cs*s over
    (tau_c<=tau_b on candidate grid, integer s). s=0 allowed iff no escalation."""
    A = pi*(kappa - fC)*rho*dist.Gu - (1-pi)*dC*dist.Ga
    prefA = np.minimum.accumulate(A)
    Bdisp = pi*fC*rho*dist.Gu + (1-pi)*dC*dist.Ga
    disp = prefA + Bdisp                              # per-query dispatch, best tau_c per j
    phi = pi*dist.Eu + (1-pi)*dist.Ea                 # escalated fraction at tau_b=T[j]
    smax = int(math.ceil(lam*phi.max()/mu)) + s_extra
    best = dict(J=math.inf)
    for j in range(len(dist.T)):
        lam_esc = lam*phi[j]
        base = lam*disp[j]
        if lam_esc <= 0:
            J = base                                   # no desk run
            if J < best["J"]:
                best = dict(J=J, j=j, s=0, lam_esc=0.0)
            continue
        s_lo = int(math.floor(lam_esc/mu)) + 1
        prev = math.inf; rise = 0
        for s in range(s_lo, smax+1):
            J = base + h*Lq(s, lam_esc, mu) + cs*s
            if J < best["J"]:
                best = dict(J=J, j=j, s=s, lam_esc=float(lam_esc))
            if J > prev:
                rise += 1
                if rise >= 3: break                     # convex in s: stop after rises
            else:
                rise = 0
            prev = J
    j = best["j"]
    # recover tau_c index
    i = int(np.argmin(np.where(np.arange(len(A)) <= j, A, np.inf)))
    best.update(i=i, tau_c=float(dist.T[i]), tau_b=float(dist.T[j]),
                esc_frac=float(phi[j]), dispatch_pq=float(disp[j]))
    return best

def desk_eval(dist, i, j, s, lam, mu, h, cs, rho, kappa, fC, dC, pi=PI):
    disp = (pi*(kappa*rho*dist.Gu[i] + fC*rho*(dist.Gu[j]-dist.Gu[i]))
            + (1-pi)*dC*(dist.Ga[j]-dist.Ga[i]))
    lam_esc = lam*(pi*dist.Eu[j] + (1-pi)*dist.Ea[j])
    if lam_esc <= 0:
        return lam*disp + (cs*s if s else 0.0)
    if s == 0 or lam_esc >= s*mu:
        return math.inf
    return lam*disp + h*Lq(s, lam_esc, mu) + cs*s

def desk_naive(dist, s, lam, mu, h, cs, rho, kappa, fC, dC, pi=PI):
    """Naive dashboard calibration (Cor. 1): the operator escalates an additional
    score slice while its per-item dispatch saving exceeds the AVERAGE delay price
    h*Wq(s, .) at the resulting load (and the load stays stable). Greedy from the
    top of the score axis; deterministic (no fixed-point cycling on atom-heavy
    empirical distributions). The optimal rule prices at the marginal externality
    h*dLq/dLambda > h*Wq, so the naive rule stops later: Lam_naive >= Lam_opt."""
    A = pi*(kappa - fC)*rho*dist.Gu - (1-pi)*dC*dist.Ga
    prefA = np.minimum.accumulate(A)
    Bdisp = pi*fC*rho*dist.Gu + (1-pi)*dC*dist.Ga
    disp = prefA + Bdisp                                # per-query dispatch, best tau_c
    phi = pi*dist.Eu + (1-pi)*dist.Ea
    n = len(dist.T)
    j = n - 1                                           # escalate nothing
    while j > 0:
        dphi = phi[j-1] - phi[j]
        if dphi <= 0:
            j -= 1; continue
        new_lam = lam*phi[j-1]
        if new_lam >= s*mu:                             # would saturate the fixed desk
            break
        saving_per_item = (disp[j] - disp[j-1]) / dphi
        if saving_per_item >= h*Wq(s, new_lam, mu):
            j -= 1
        else:
            break
    i = int(np.argmin(np.where(np.arange(len(A)) <= j, A, np.inf)))
    J = desk_eval(dist, i, j, s if phi[j] > 0 else 0, lam, mu, h, cs, rho, kappa, fC, dC)
    return dict(j=j, i=i, tau_b=float(dist.T[j]), lam_esc=float(lam*phi[j]), J=float(J))

# ---------------------------------------------------------------- QED machinery
def hw_alpha(b):
    from math import erf, exp, pi as MPI, sqrt
    Phi = 0.5*(1+erf(b/sqrt(2))); phi = exp(-b*b/2)/sqrt(2*MPI)
    return 1.0/(1.0 + b*Phi/phi)

def beta_star(cs, h):
    grid = np.linspace(1e-3, 20, 4001)
    g = cs*grid + h*np.array([hw_alpha(b) for b in grid])/grid
    k = int(np.argmin(g))
    lo, hi = grid[max(k-1,0)], grid[min(k+1,len(grid)-1)]
    for _ in range(80):                                 # golden-ish refine
        m1, m2 = lo + (hi-lo)/3, hi - (hi-lo)/3
        f = lambda b: cs*b + h*hw_alpha(b)/b
        if f(m1) < f(m2): hi = m2
        else: lo = m1
    b = 0.5*(lo+hi)
    return float(b), float(cs*b + h*hw_alpha(b)/b)

def desk_qed(dist, lam, mu, h, cs, rho, kappa, fC, dC, pi=PI):
    """Square-root-staffing approximation: choose tau_b minimizing
    lam*dispatch + cs*R + sqrt(R)*g(beta*), staff s = ceil(R + beta* sqrt(R)),
    then evaluate that policy under EXACT Erlang-C."""
    bstar, gstar = beta_star(cs, h)
    A = pi*(kappa - fC)*rho*dist.Gu - (1-pi)*dC*dist.Ga
    prefA = np.minimum.accumulate(A)
    Bdisp = pi*fC*rho*dist.Gu + (1-pi)*dC*dist.Ga
    phi = pi*dist.Eu + (1-pi)*dist.Ea
    R = lam*phi/mu
    Japprox = lam*(prefA + Bdisp) + cs*R + np.sqrt(R)*gstar
    j = int(np.argmin(Japprox))
    i = int(np.argmin(np.where(np.arange(len(A)) <= j, A, np.inf)))
    Rj = float(R[j])
    s = 0 if Rj == 0 else max(1, int(math.ceil(Rj + bstar*math.sqrt(Rj))))
    while s > 0 and lam*phi[j] >= s*mu:
        s += 1
    Jex = desk_eval(dist, i, j, s, lam, mu, h, cs, rho, kappa, fC, dC)
    return dict(beta_star=bstar, j=j, i=i, tau_b=float(dist.T[j]), s=s, R=Rj,
                J_approx=float(Japprox[j]), J_exact_of_qed_policy=float(Jex))

# ---------------------------------------------------------------- MLR check
def mlr_check(u, a, nbins=10):
    """Empirical likelihood-ratio monotonicity over pooled-quantile bins."""
    pooled = np.concatenate([u, a])
    qs = np.quantile(pooled, np.linspace(0, 1, nbins+1))
    edges = np.unique(qs)
    cu, _ = np.histogram(u, bins=edges)
    ca, _ = np.histogram(a, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = (cu/len(u)) / (ca/len(a))
    lr = np.where((cu == 0) & (ca == 0), np.nan, lr)
    lr = np.where((ca == 0) & (cu > 0), np.inf, lr)
    viol, viol_mass = [], 0.0
    finite_seq = [x for x in lr if not np.isnan(x)]
    keep = [k for k in range(len(lr)) if not np.isnan(lr[k])]
    for t in range(1, len(finite_seq)):
        if finite_seq[t] < finite_seq[t-1] - 1e-12:
            k = keep[t]
            viol.append(dict(bin=k, lr_prev=float(finite_seq[t-1]),
                             lr=float(finite_seq[t])))
            viol_mass += float((cu[k]+ca[k]) / (len(u)+len(a)))
    atom = max(float(np.max(np.unique(u, return_counts=True)[1]))/len(u),
               float(np.max(np.unique(a, return_counts=True)[1]))/len(a))
    return dict(n_bins_used=len(edges)-1,
                bin_edges=[float(x) for x in edges],
                count_u=cu.tolist(), count_a=ca.tolist(),
                lr=[None if np.isnan(x) else (1e18 if np.isinf(x) else float(x)) for x in lr],
                monotone=len(viol) == 0, violations=viol,
                violation_traffic_mass=float(viol_mass),
                largest_atom_mass=atom)

# ================================================================ RUN
def main():
    scores = load_scores()
    u_q, a_q = scores[("param", "block_gate_score")]
    qwen_emp = Dist.from_samples(u_q, a_q)
    sep_gap = [float(a_q.max()), float(u_q.min())]      # empirical separation gap

    out = dict(_meta=dict(
        script="w2_numerics.py", pi=PI,
        score_source="results/cast_stage_items.jsonl (param loop, block_gate_score; n=73/73)",
        llama_note=("per-item Llama gate scores are not in the replication package; "
                    "Llama uses the two-atom operating-point representation E_u=69/73, E_a=2/73"),
        cp_bounds=dict(dC_0_of_73_upper=DC_UB, Eu_73_of_73_lower=EU_LB, Ea_0_of_73_upper=EA_UB,
                       fC_qwen_6_of_73=list(cp_interval(6, 73)),
                       kappa_qwen_60_of_72=list(cp_interval(60, 72)),
                       fC_llama_50_of_73=list(cp_interval(50, 73)),
                       Eu_llama_69_of_73=list(cp_interval(69, 73)),
                       Ea_llama_2_of_73=list(cp_interval(2, 73))),
        empirical_separation_gap=dict(answerable_max=sep_gap[0], unanswerable_min=sep_gap[1],
                                      perfectly_separated=bool(sep_gap[0] < sep_gap[1]))))

    # ---------------- 1. static frontier rho* --------------------------------
    frontier = {}
    cesc_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0, 2.0, math.inf]
    for model, prm, dist_measured in [
        ("qwen_empirical", QWEN, qwen_emp),
        ("qwen_two_atom_measured", QWEN, Dist.two_atom(1.0, 0.0)),
        ("qwen_two_atom_cp", QWEN, Dist.two_atom(EU_LB, EA_UB)),
        ("llama_two_atom_measured", LLAMA, Dist.two_atom(69/73, 2/73)),
        ("llama_two_atom_cp", LLAMA, Dist.two_atom(cp_interval(69,73)[0], cp_interval(2,73)[1])),
    ]:
        for dC, dtag in [(0.0, "dC=0"), (DC_UB, f"dC={DC_UB:.4f}")]:
            row = {}
            for ce in cesc_grid:
                rs = rho_star(dist_measured, ce, prm["kappa"], prm["fC"], dC)
                row[f"cesc={ce}"] = rs if math.isfinite(rs) else "inf"
            frontier[f"{model}|{dtag}"] = row
    # closed forms the paper's numbers correspond to
    frontier["_closed_forms"] = dict(
        qwen_escalation_free_1_over_fC=73/6,
        qwen_escalation_free_dC_ub=(1-DC_UB)/(6/73),
        llama_escalation_free_1_over_fC=73/50,
        llama_escalation_free_dC_ub=(1-DC_UB)/(50/73),
        qwen_1_over_fC_at_fC_CI=[1/cp_interval(6,73)[1], 1/cp_interval(6,73)[0]],
        llama_1_over_fC_at_fC_CI=[1/cp_interval(50,73)[1], 1/cp_interval(50,73)[0]])
    # Pi* monotonicity witness (verification aid)
    rgrid = np.logspace(-1, 4, 60)
    frontier["_pi_star_curve_qwen_cesc0.05_dC0"] = [
        [float(r), static_best(qwen_emp, r, 0.05, QWEN["kappa"], QWEN["fC"], 0.0)["cost"]]
        for r in rgrid]
    out["static_frontier"] = frontier

    # ---------------- 2. boundary / interiority ------------------------------
    inter = {}
    for dC, dtag in [(0.0, "dC=0"), (DC_UB, f"dC={DC_UB:.4f}")]:
        for rho in [1.5, 5.0, 12.0]:
            for ce in [0.01, 0.05, 0.1, 1.0]:
                b = static_best(qwen_emp, rho, ce, QWEN["kappa"], QWEN["fC"], dC)
                Tj = b["tau_b"]; Ti = b["tau_c"]
                tau_c_boundary = (b["i"] == 0) or (b["i"] == len(qwen_emp.T)-1)
                # tau_b interior needs empirical mass on both sides AND indifference;
                # in-gap = between answerable max and unanswerable min (no marginal item)
                in_gap = (sep_gap[0] < Tj <= sep_gap[1])
                tau_b_boundary = (b["j"] == 0) or (b["j"] == len(qwen_emp.T)-1)
                inter[f"{dtag}|rho={rho}|cesc={ce}"] = dict(
                    tau_c=Ti, tau_b=Tj,
                    answer_region_empty=bool(b["i"] == 0),
                    answer_region_all=bool(b["i"] == len(qwen_emp.T)-1),
                    escalate_region_empty=bool(b["j"] == len(qwen_emp.T)-1),
                    escalate_region_all=bool(b["j"] == 0),
                    tau_b_in_separation_gap=bool(in_gap),
                    interior=bool((not tau_c_boundary) and (not tau_b_boundary)
                                  and (not in_gap)),
                    Gu_at_tau_c=b["Gu_c"], Eu_at_tau_b=b["Eu_b"], Ea_at_tau_b=b["Ea_b"],
                    per_query_cost=b["cost"])
    # A7 / regime restriction behavior
    out["boundary_treatment"] = dict(
        dC_cp_upper=DC_UB,
        A7_regime=dict(
            note="A7 is fC*rho > dC; regime restriction of Prop. 8 sweep is rho > dC/fC",
            qwen_dC0="A7 holds for every rho>0 (restriction vacuous, as the paper states)",
            qwen_dC_ub_rho_min=DC_UB/(6/73),
            llama_dC_ub_rho_min=DC_UB/(50/73)),
        interiority=inter,
        static_rule_posterior_at_tau_c=dict(
            note="eq (11): p(tau_c*) = dC/((kappa-fC)rho + dC); 0 at dC=0 -> tau_c* boundary",
            qwen_dC0=0.0,
            qwen_dC_ub_rho12=DC_UB/((QWEN["kappa"]-QWEN["fC"])*12 + DC_UB)))

    # gate-rate degeneracy: does anything flip at CP bounds? compare frontiers
    out["gate_rate_flip_check"] = dict(
        qwen_measured=out["static_frontier"]["qwen_two_atom_measured|dC=0"],
        qwen_cp=out["static_frontier"]["qwen_two_atom_cp|dC=0"],
        comment="computed above; see w2_numerics.md for the reading")

    # ---------------- 3. MLR check -------------------------------------------
    mlr = {}
    for (loop, key), (u, a) in scores.items():
        mlr[f"{loop}|{key}"] = mlr_check(u, a, nbins=10)
    out["mlr"] = mlr

    # ---------------- 4. worked desk example ---------------------------------
    MU = 12.0
    desk = dict(_params=dict(mu_per_hour=MU, pi=PI, model="Qwen empirical scores",
                             kappa=QWEN["kappa"], fC=QWEN["fC"], dC=0.0,
                             h_units="fabrication-costs/hour (h_numeraire = h_f * rho)",
                             note="illustrative parameters, stated as such"))
    cells = []
    for rho in [1.5, 5.0, 12.0]:
        for h_f in [0.1, 0.5, 1.0]:
            h = h_f*rho
            for cs in [2.0, 10.0, 50.0]:
                for lam in [10.0, 100.0, 1000.0]:
                    ex = desk_exact(qwen_emp, lam, MU, h, cs, rho, QWEN["kappa"], QWEN["fC"], 0.0)
                    cell = dict(rho=rho, h_fabcost_per_hr=h_f, h=h, cs=cs, lam=lam,
                                exact=dict(J=ex["J"], s=ex["s"], esc_frac=ex["esc_frac"],
                                           lam_esc=ex["lam_esc"], tau_b=ex["tau_b"],
                                           tau_c=ex["tau_c"]))
                    if ex["s"] >= 1:
                        nv = desk_naive(qwen_emp, ex["s"], lam, MU, h, cs, rho,
                                        QWEN["kappa"], QWEN["fC"], 0.0)
                        cell["naive"] = dict(J=nv["J"], lam_esc=nv["lam_esc"], tau_b=nv["tau_b"])
                        cell["wedge"] = dict(cost=float(nv["J"]-ex["J"]),
                                             over_escalation=float(nv["lam_esc"]-ex["lam_esc"]))
                    qd = desk_qed(qwen_emp, lam, MU, h, cs, rho, QWEN["kappa"], QWEN["fC"], 0.0)
                    gap = ((qd["J_exact_of_qed_policy"] - ex["J"])/ex["J"]
                           if ex["J"] > 0 else 0.0)
                    cell["qed"] = dict(beta_star=qd["beta_star"], s=qd["s"], R=qd["R"],
                                       J_exact_of_qed_policy=qd["J_exact_of_qed_policy"],
                                       rel_gap=float(gap))
                    cells.append(cell)
    desk["cells"] = cells
    out["desk"] = desk

    # ---------------- 4b. wedge demonstration on an overlapping ROC ----------
    # At the calibrated parametric scores the ROC is perfectly separated (two-atom
    # in effect): optimal and naive both escalate exactly the unanswerable support,
    # and Cor. 1's wedge is degenerate at zero. The grounded arm's gate ROC overlaps,
    # giving a genuine interior margin. Rates measured on the grounded arm itself
    # (agentic_loop_cast_sonnet.json): kappa=15/108, fC=12/108, dC=2/108.
    u_g, a_g = scores[("grounded", "block_gate_score")]
    grd = Dist.from_samples(u_g, a_g)
    GK, GF, GD = 15/108, 12/108, 2/108
    wcells = []
    for rho in [5.0, 12.0]:
        for h_f in [0.1, 0.5, 1.0]:
            h = h_f*rho
            for cs in [2.0, 10.0]:
                for lam in [100.0, 1000.0]:
                    ex = desk_exact(grd, lam, MU, h, cs, rho, GK, GF, GD)
                    cell = dict(rho=rho, h_fabcost_per_hr=h_f, cs=cs, lam=lam,
                                exact=dict(J=ex["J"], s=ex["s"], lam_esc=ex["lam_esc"],
                                           esc_frac=ex["esc_frac"], tau_b=ex["tau_b"]))
                    if ex["s"] >= 1:
                        nv = desk_naive(grd, ex["s"], lam, MU, h, cs, rho, GK, GF, GD)
                        cell["naive"] = dict(J=nv["J"], lam_esc=nv["lam_esc"])
                        cell["wedge"] = dict(cost=float(nv["J"]-ex["J"]),
                                             over_escalation=float(nv["lam_esc"]-ex["lam_esc"]),
                                             rel=float((nv["J"]-ex["J"])/ex["J"]))
                    wcells.append(cell)
    out["wedge_grounded"] = dict(
        _params=dict(kappa=GK, fC=GF, dC=GD, mu=MU,
                     roc="grounded-arm block gate scores (overlapping ROC), n=108/108",
                     note="grounded-arm measured technology rates; desk parameters illustrative"),
        cells=wcells)

    # ---------------- 4c. de-automation frontier lambda^dagger ---------------
    # fixed-s frontier, accounting convention of Sec. 4.2: automated side always
    # charged cs*s at the fixed s; baseline J0 = lam*(1-pi).
    def lam_dagger(dist, s, mu, h, cs, rho, kappa, fC, dC, lam_max=1e6):
        A = PI*(kappa - fC)*rho*dist.Gu - (1-PI)*dC*dist.Ga
        prefA = np.minimum.accumulate(A)
        Bdisp = PI*fC*rho*dist.Gu + (1-PI)*dC*dist.Ga
        phi = PI*dist.Eu + (1-PI)*dist.Ea
        def Jstar(lam):
            with np.errstate(over="ignore"):
                cong = np.array([h*Lq(s, lam*p, mu) if lam*p < s*mu else math.inf
                                 for p in phi])
            return float(np.min(lam*(prefA + Bdisp) + cong)) + cs*s
        # H-fr1: escalation-free per-query cost vs 1-pi (tau_b = +inf, dispatch only)
        cAC = float(prefA[-1] + Bdisp[-1])
        J0 = lambda lam: lam*(1-PI)
        lam1 = None
        for lam in np.logspace(-1, 6, 200):
            if Jstar(lam) < J0(lam):
                lam1 = float(lam); break
        if lam1 is None:
            return dict(cAC=cAC, H_fr1=bool(cAC > 1-PI), H_fr2=False, lam_dagger="never_automate")
        lo, hi = lam1, None
        for lam in np.logspace(math.log10(lam1), 8, 400):
            if Jstar(lam) >= J0(lam):
                hi = float(lam); break
            lo = float(lam)
        if hi is None:
            return dict(cAC=cAC, H_fr1=bool(cAC > 1-PI), H_fr2=True, lam1=lam1,
                        lam_dagger="inf")
        for _ in range(120):
            mid = math.sqrt(lo*hi)
            if Jstar(mid) < J0(mid): lo = mid
            else: hi = mid
        ld = float(math.sqrt(lo*hi))
        # (H-fr3) check: does automation re-dominate above the first crossing?
        reentry = None
        for lam in np.logspace(math.log10(ld)+0.05, 8, 300):
            if Jstar(lam) < J0(lam):
                reentry = float(lam); break
        return dict(cAC=cAC, H_fr1=bool(cAC > 1-PI), H_fr2=True, lam1=lam1,
                    lam_dagger=ld, reentry_lambda=reentry,
                    single_crossing_H_fr3=bool(reentry is None))

    front = {}
    for rho in [1.5, 5.0, 12.0, 15.0]:
        for dC, dtag in [(0.0, "dC=0"), (DC_UB, f"dC={DC_UB:.4f}")]:
            for dist, gtag in [(qwen_emp, "empirical"),
                               (Dist.two_atom(EU_LB, EA_UB), "cp_gate")]:
                for cs in [2.0, 10.0]:
                    for s in [1, 4, 12]:
                        r = lam_dagger(dist, s, MU, 0.5*rho, cs, rho,
                                       QWEN["kappa"], QWEN["fC"], dC)
                        front[f"rho={rho}|{dtag}|{gtag}|cs={cs}|s={s}"] = r
    out["lambda_dagger"] = dict(
        _params=dict(h="0.5*rho per hour", cs="{2,10}", mu=MU,
                     convention="cs*s charged to every automated policy at fixed s"),
        results=front)

    with open(os.path.join(RES, "w2_numerics.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote results/w2_numerics.json")

    # console headline
    print("\nrho* (Prop 8 Pi*, Qwen empirical, dC=0):")
    for ce in cesc_grid:
        print(f"  c0={ce}: {frontier['qwen_empirical|dC=0'][f'cesc={ce}']}")
    print("closed forms:", frontier["_closed_forms"])

if __name__ == "__main__":
    main()
