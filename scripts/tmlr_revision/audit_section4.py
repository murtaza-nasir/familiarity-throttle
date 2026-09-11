#!/usr/bin/env python3
"""Section 4 audit: rebuild every ladder number from replication_package/data with explicit n, CIs, and p-values,
under the primary judge (Sonnet) and the alternative judge (GPT-5.6) where verdicts exist."""
import json, numpy as np, pandas as pd
from scipy import stats
D = "replication_package/data"
J = lambda f: [json.loads(l) for l in open(f"{D}/{f}") if l.strip()]
ents = json.load(open(f"{D}/graded_tier_entities.json"))["entities"]
excl = {e["entity"] for e in ents["S2"] if e.get("recheck_verdict") == "excluded"}
batch = {e["entity"]: e.get("batch", 1) for e in ents["S2"]}
son = {}; gpt = {}
for f in ["graded_tier_verdicts_sonnet.jsonl","graded_tier_s2b_verdicts.jsonl","graded_tier_s1s5_restyled_verdicts.jsonl"]:
    for r in J(f):
        if r["verdict"] in ("COMMIT","DEFLECT","FLAG"): son[r["id"]] = r["verdict"]
for f in ["graded_tier_newarms_verdicts_OPENAI.jsonl","graded_tier_qwenladder_missing_verdicts_OPENAI.jsonl"]:
    for r in J(f):
        if r["verdict"] in ("COMMIT","DEFLECT","FLAG"): gpt[r["id"]] = r["verdict"]
rows = []
for g in J("graded_tier_gens.jsonl"):
    if g["stratum"] in ("S2","S3","S4") and not (g["stratum"]=="S2" and g["entity"] in excl):
        rows.append(dict(id=g["id"], stratum=g["stratum"], entity=g["entity"], probe=g["probe_score"], gen=g["gen"],
                         batch=("b1" if g["stratum"]=="S2" else None)))
for g in J("graded_tier_s1s5_restyled_gens.jsonl"):
    rows.append(dict(id=g["id"], stratum=g["stratum"], entity=g["entity"], probe=g["probe_score"], gen=g["gen"], batch=None))
for g in J("graded_tier_s2b_gens.jsonl"):
    rows.append(dict(id=g["id"], stratum="S2", entity=g["entity"], probe=g["probe_score"], gen=g["gen"], batch="b2"))
df = pd.DataFrame(rows)
df["son"] = df.id.map(son); df["gpt"] = df.id.map(gpt)
print("missing sonnet verdicts:", int(df.son.isna().sum()), " missing gpt:", int(df.gpt.isna().sum()))
df = df[df.son.notna()].copy()
ORDER = ["S1","S2","S3","S4","S5"]
def bin_ci(k,n):
    lo,hi = stats.beta.ppf([0.025,0.975],[k,k+1],[n-k+1,n-k]); return (0 if k==0 else lo, 1 if k==n else hi)
def boot_entity_mean(sub, col, B=5000, seed=0):
    em = sub.groupby("entity")[col].mean().values; rng=np.random.default_rng(seed)
    bs = rng.choice(em, (B, len(em))).mean(1); return float(em.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))
out = {"strata": {}, "tests": {}, "judge_sensitivity": {}}
print(f"\n{'S':<3}{'items':>6}{'ents':>6}{'probe_mean':>11}{'[boot CI ent]':>18}{'median':>8}{'IQR':>16} | {'commit':>7}{'[CI]':>15}{'deflect':>8}{'flag':>6}{'[CI]':>15}")
for s in ORDER:
    sub = df[df.stratum==s]; n=len(sub); ne=sub.entity.nunique()
    pm, plo, phi = boot_entity_mean(sub, "probe")
    q1,med,q3 = np.percentile(sub.probe,[25,50,75])
    rec = {"n_items": n, "n_entities": ne, "probe_mean_items": float(sub.probe.mean()), "probe_mean_entity_boot": [pm,plo,phi],
           "probe_median": float(med), "probe_iqr": [float(q1),float(q3)]}
    for jname, col in [("sonnet","son"),("gpt","gpt")]:
        v = sub[col].dropna(); m=len(v)
        if m==0: continue
        r = {}
        for cat in ["COMMIT","DEFLECT","FLAG"]:
            k=int((v==cat).sum()); r[cat.lower()] = {"k":k,"n":m,"rate":k/m,"ci":list(bin_ci(k,m))}
        # entity-level means
        vv = sub.loc[v.index]; r["flag_entity_mean"] = float((vv[col]=="FLAG").groupby(vv.entity).mean().mean())
        r["commit_entity_mean"] = float((vv[col]=="COMMIT").groupby(vv.entity).mean().mean())
        rec[jname] = r
    out["strata"][s] = rec
    c=rec["sonnet"]["commit"]; f=rec["sonnet"]["flag"]
    print(f"{s:<3}{n:>6}{ne:>6}{sub.probe.mean():>11.3f}{f'[{plo:.3f},{phi:.3f}]':>18}{med:>8.3f}{f'[{q1:.3f},{q3:.3f}]':>16} | {c['rate']:>7.3f}{f'[{c['ci'][0]:.2f},{c['ci'][1]:.2f}]':>15}{rec['sonnet']['deflect']['rate']:>8.3f}{f['rate']:>6.3f}{f'[{f['ci'][0]:.2f},{f['ci'][1]:.2f}]':>15}")
    if s=="S2":
        for b in ["b1","b2"]:
            sb = sub[sub.batch==b]; print(f"   S2 {b}: items {len(sb)} ents {sb.entity.nunique()} probe {sb.probe.mean():.3f} commit {(sb.son=='COMMIT').mean():.3f} flag {(sb.son=='FLAG').mean():.3f}")
            out["strata"]["S2"][b] = {"n_items":len(sb),"n_entities":int(sb.entity.nunique()),"probe_mean":float(sb.probe.mean()),"commit":float((sb.son=='COMMIT').mean()),"flag":float((sb.son=='FLAG').mean())}
# entity means
em = df.groupby(["stratum","entity"]).agg(probe=("probe","mean"), flag=("son", lambda v:(v=="FLAG").mean()), commit=("son", lambda v:(v=="COMMIT").mean())).reset_index()
def mw(a,b): 
    u,p = stats.mannwhitneyu(a,b,alternative="two-sided"); return {"n":[len(a),len(b)],"U":float(u),"p":float(p)}
E = lambda s: em[em.stratum==s]
T = out["tests"]
T["probe_S3_vs_S4_MW_entity"] = mw(E("S3").probe, E("S4").probe)
T["probe_S2_vs_S1_MW_entity"] = mw(E("S2").probe, E("S1").probe)
T["probe_S2_vs_S5_MW_entity"] = mw(E("S2").probe, E("S5").probe)
T["probe_S2_vs_S4_MW_entity"] = mw(E("S2").probe, E("S4").probe)
rank = em.stratum.map({s:i for i,s in enumerate(ORDER)})
rho,p = stats.spearmanr(rank, em.flag); T["flag_spearman_entity"] = {"rho":float(rho),"p":float(p),"n_entities":len(em)}
ri = df.stratum.map({s:i for i,s in enumerate(ORDER)}); rho2,p2 = stats.spearmanr(ri, df.son=="FLAG"); T["flag_spearman_item"] = {"rho":float(rho2),"p":float(p2),"n_items":len(df)}
def fisher(s1,s2,cat):
    a=df[df.stratum==s1].son; b=df[df.stratum==s2].son
    tab=[[int((a==cat).sum()),int((a!=cat).sum())],[int((b==cat).sum()),int((b!=cat).sum())]]
    o,p=stats.fisher_exact(tab); return {"table":tab,"odds_ratio":float(o),"p":float(p)}
T["commit_S1_vs_S2_fisher_item"] = fisher("S1","S2","COMMIT")
T["flag_S3_vs_S4_fisher_item"] = fisher("S3","S4","FLAG")
T["flag_S2_vs_S3_fisher_item"] = fisher("S2","S3","FLAG")
T["flag_S4_vs_S5_fisher_item"] = fisher("S4","S5","FLAG")
T["flag_S1_vs_S2_fisher_item"] = fisher("S1","S2","FLAG")
# entity-level MW on flag rates for adjacent rungs
for a,b in [("S1","S2"),("S2","S3"),("S3","S4"),("S4","S5")]:
    T[f"flag_{a}_vs_{b}_MW_entity"] = mw(E(a).flag, E(b).flag)
k=int((df[df.stratum=="S2"].son=="COMMIT").sum()); n=int((df.stratum=="S2").sum()); T["S2_commit_count"] = {"k":k,"n":n,"ci":list(bin_ci(k,n))}
# judge sensitivity: agreement + rates under gpt for items with both
both = df[df.gpt.notna()]
T["judge_agreement_ladder"] = {"n":len(both),"agreement":float((both.son==both.gpt).mean()), "kappa": None}
from sklearn.metrics import cohen_kappa_score
T["judge_agreement_ladder"]["kappa"] = float(cohen_kappa_score(both.son, both.gpt))
conf = pd.crosstab(both.son, both.gpt); print("\nSonnet(rows) x GPT(cols) confusion:\n", conf)
out["judge_sensitivity"]["confusion"] = conf.to_dict()
emg = both.groupby(["stratum","entity"]).agg(flag=("gpt", lambda v:(v=="FLAG").mean())).reset_index()
rk = emg.stratum.map({s:i for i,s in enumerate(ORDER)}); rho3,p3 = stats.spearmanr(rk, emg.flag)
out["judge_sensitivity"]["flag_spearman_entity_gpt"] = {"rho":float(rho3),"p":float(p3),"n":len(emg)}
a=both[both.stratum=="S1"].gpt; b=both[both.stratum=="S2"].gpt
o,p=stats.fisher_exact([[int((a=="COMMIT").sum()),int((a!="COMMIT").sum())],[int((b=="COMMIT").sum()),int((b!="COMMIT").sum())]])
out["judge_sensitivity"]["commit_S1_vs_S2_fisher_gpt"] = {"p":float(p),"or":float(o)}
print("\nTESTS:"); print(json.dumps(T, indent=1))
print("\nJUDGE SENS:", json.dumps({k:v for k,v in out["judge_sensitivity"].items() if k!="confusion"}, indent=1))
# name-shape lexical features (tokenizer-free part)
import re
df["n_words"] = df.entity.str.split().str.len(); df["has_digit"] = df.entity.str.contains(r"\d"); df["n_chars"]=df.entity.str.len()
df["has_nonascii"] = df.entity.apply(lambda s: any(ord(c)>127 for c in s))
print("\nlexical by stratum:\n", df.groupby("stratum")[["n_words","n_chars","has_digit"]].mean().round(2))
df.to_csv("tmlr_revision/ladder_items.csv", index=False)
json.dump(out, open("tmlr_revision/audit_section4.json","w"), indent=1)
print("wrote tmlr_revision/audit_section4.json + ladder_items.csv")
