#!/usr/bin/env python3
"""Matched-probe S2-vs-S4 dissociation: entity-clustered tests.

Primary: GEE logistic commit ~ probe + stratum, exchangeable correlation
within entity, over all S2 (clean) and S4 items.
Secondary: Cochran-Mantel-Haenszel within probe quintiles; permutation
test on entity means within the probe>0.5 slice.

Inputs (replication_package/data/): graded_tier_entities.json,
graded_tier_gens.jsonl + graded_tier_verdicts_sonnet.jsonl,
graded_tier_s2b_gens.jsonl + graded_tier_s2b_verdicts.jsonl.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

ents = json.load(open(f"{DATA}/graded_tier_entities.json"))["entities"]["S2"]
excl = {e["entity"] for e in ents if e.get("recheck_verdict") == "excluded"}

rows = []
gens = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/graded_tier_gens.jsonl")}
verd = {}
for l in open(f"{DATA}/graded_tier_verdicts_sonnet.jsonl"):
    r = json.loads(l)
    if r["verdict"] in ("COMMIT", "DEFLECT", "FLAG"):
        verd[r["id"]] = r["verdict"]
for i, g in gens.items():
    if g["stratum"] in ("S2", "S4") and not (g["stratum"] == "S2" and g["entity"] in excl) and i in verd:
        rows.append((g["stratum"], g["entity"], g["probe_score"], verd[i] == "COMMIT"))
bg = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/graded_tier_s2b_gens.jsonl")}
for l in open(f"{DATA}/graded_tier_s2b_verdicts.jsonl"):
    r = json.loads(l)
    if r["verdict"] in ("COMMIT", "DEFLECT", "FLAG"):
        g = bg[r["id"]]
        rows.append(("S2", g["entity"], g["probe_score"], r["verdict"] == "COMMIT"))

df = pd.DataFrame(rows, columns=["stratum", "entity", "probe", "commit"])
df["commit"] = df["commit"].astype(int)
df["s4"] = (df["stratum"] == "S4").astype(int)
print(f"items: {len(df)} (S2 {int((df.s4==0).sum())}, S4 {int((df.s4==1).sum())})")

m = smf.gee("commit ~ probe + s4", "entity", df, family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable()).fit()
print(f"GEE: S4 coef {m.params['s4']:.3f} (OR {np.exp(m.params['s4']):.2f}), p = {m.pvalues['s4']:.4f}")

bins = np.quantile(df["probe"], [0, .2, .4, .6, .8, 1.0])
df["bin"] = np.digitize(df["probe"], bins[1:-1])
tabs = []
for _, gd in df.groupby("bin"):
    tabs.append(np.array([[((gd.s4 == 1) & (gd.commit == 1)).sum(), ((gd.s4 == 1) & (gd.commit == 0)).sum()],
                          [((gd.s4 == 0) & (gd.commit == 1)).sum(), ((gd.s4 == 0) & (gd.commit == 0)).sum()]], float))
st = StratifiedTable(tabs)
print(f"CMH (probe quintiles): pooled OR {st.oddsratio_pooled:.2f}, p = {st.test_null_odds().pvalue:.4f}")

hi = df[df.probe > 0.5]
em = hi.groupby(["stratum", "entity"])["commit"].mean().reset_index()
obs = em[em.stratum == "S4"]["commit"].mean() - em[em.stratum == "S2"]["commit"].mean()
rng = np.random.default_rng(0)
lab = em["stratum"].values.copy()
cm = em["commit"].values
diffs = []
for _ in range(20000):
    rng.shuffle(lab)
    diffs.append(cm[lab == "S4"].mean() - cm[lab == "S2"].mean())
p = (np.sum(np.array(diffs) >= obs) + 1) / 20001
print(f"permutation on entity means (probe>0.5): diff {obs:.3f}, one-sided p = {p:.4f}")

out = {"n_items": len(df),
       "gee": {"or": float(np.exp(m.params["s4"])), "p": float(m.pvalues["s4"])},
       "cmh": {"or": float(st.oddsratio_pooled), "p": float(st.test_null_odds().pvalue)},
       "permutation_entity_means": {"diff": float(obs), "p_one_sided": float(p)}}
json.dump(out, open(os.path.join(HERE, "..", "results", "matched_probe_tests.json"), "w"), indent=1)
print("wrote results/matched_probe_tests.json")
