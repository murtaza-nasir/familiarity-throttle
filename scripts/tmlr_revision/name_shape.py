#!/usr/bin/env python3
"""TMLR revision C2(i): does name shape explain the ladder inversion? Entity-level analysis of probe score against
tokenization/lexical features of the entity name (Qwen3-8B tokenizer), within and across strata."""
import json, re, numpy as np, pandas as pd
from transformers import AutoTokenizer
import statsmodels.formula.api as smf
from scipy import stats
D = "$HN_ROOT/data/abstention_full"
tok = AutoTokenizer.from_pretrained("$MODEL_ROOT/Qwen-Qwen3-8B")
df = pd.read_csv(f"{D}/ladder_items.csv")
em = df.groupby(["stratum", "entity"]).agg(probe=("probe", "mean"), flag=("son", lambda v: (v == "FLAG").mean()), commit=("son", lambda v: (v == "COMMIT").mean())).reset_index()
def feats(name):
    ids = tok(" " + name, add_special_tokens=False)["input_ids"]; toks = [tok.decode([i]) for i in ids]
    words = name.split()
    wordlike = [t for t in toks if re.fullmatch(r" ?[A-Za-z]{3,}", t)]
    return dict(n_tok=len(ids), n_words=len(words), tok_per_word=len(ids)/max(1, len(words)), chars_per_tok=len(name)/len(ids),
                frac_wordlike=len(wordlike)/len(ids), has_digit=int(bool(re.search(r"\d", name))), n_chars=len(name),
                frag=int(len(ids) > len(words)))  # fragmented: more tokens than words
F = pd.DataFrame([feats(e) for e in em.entity]); em = pd.concat([em, F], axis=1)
em["conventional"] = ((em.tok_per_word <= 1.0) & (em.has_digit == 0)).astype(int)
out = {"by_stratum_features": em.groupby("stratum")[["n_tok", "tok_per_word", "chars_per_tok", "frac_wordlike", "has_digit", "conventional"]].mean().round(3).to_dict()}
print(pd.DataFrame(out["by_stratum_features"]).round(3), flush=True)
# 1. within-stratum association of probe with name features
w = {}
for s, g in em.groupby("stratum"):
    r = {}
    for f in ["tok_per_word", "chars_per_tok", "frac_wordlike", "n_tok"]:
        rho, p = stats.spearmanr(g[f], g.probe); r[f] = {"rho": float(rho), "p": float(p), "n": len(g)}
    if g.conventional.nunique() > 1:
        a = g[g.conventional == 1].probe; b = g[g.conventional == 0].probe
        u, p = stats.mannwhitneyu(a, b); r["conventional_vs_odd"] = {"n_conv": len(a), "n_odd": len(b), "mean_conv": float(a.mean()), "mean_odd": float(b.mean()), "MW_p": float(p)}
    w[s] = r
out["within_stratum"] = w; print(json.dumps(w, indent=1), flush=True)
# 2. does stratum explain probe beyond name features? nested OLS on entity means
em["S"] = em.stratum
m0 = smf.ols("probe ~ tok_per_word + chars_per_tok + frac_wordlike + has_digit + n_words", em).fit()
m1 = smf.ols("probe ~ tok_per_word + chars_per_tok + frac_wordlike + has_digit + n_words + C(S)", em).fit()
m2 = smf.ols("probe ~ C(S)", em).fit()
from statsmodels.stats.anova import anova_lm
an = anova_lm(m0, m1)
out["nested_ols"] = {"R2_features_only": float(m0.rsquared), "R2_stratum_only": float(m2.rsquared), "R2_features_plus_stratum": float(m1.rsquared),
                     "F_stratum_given_features": float(an["F"].iloc[1]), "p_stratum_given_features": float(an["Pr(>F)"].iloc[1]),
                     "adjusted_stratum_means": {k: float(v) for k, v in m1.params.items() if k.startswith("C(S)")}}
print(json.dumps(out["nested_ols"], indent=1), flush=True)
# 3. the inversion within matched name shape: S3 vs S4 restricted to conventional names, and to odd names
inv = {}
for lab, mask in [("conventional", em.conventional == 1), ("odd", em.conventional == 0), ("frag0", em.frag == 0), ("frag1", em.frag == 1)]:
    a = em[(em.S == "S3") & mask].probe; b = em[(em.S == "S4") & mask].probe
    if len(a) >= 5 and len(b) >= 5:
        u, p = stats.mannwhitneyu(a, b); inv[lab] = {"n_S3": len(a), "n_S4": len(b), "mean_S3": float(a.mean()), "mean_S4": float(b.mean()), "MW_p": float(p)}
out["S3_vs_S4_within_name_shape"] = inv; print(json.dumps(inv, indent=1), flush=True)
# 4. residualized probe (remove name features) still orders strata?
em["probe_resid"] = m0.resid
out["stratum_means_residualized"] = em.groupby("S").probe_resid.mean().round(3).to_dict()
rank = em.S.map({"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4})
rho, p = stats.spearmanr(rank, em.probe_resid); out["spearman_rank_vs_resid"] = {"rho": float(rho), "p": float(p)}
a = em[em.S == "S3"].probe_resid; b = em[em.S == "S4"].probe_resid; u, p = stats.mannwhitneyu(a, b); out["S3_vs_S4_residualized_MW_p"] = float(p)
print("residualized stratum means:", out["stratum_means_residualized"], "S3vsS4 resid p", p, flush=True)
em.to_csv(f"{D}/name_shape_entities.csv", index=False)
json.dump(out, open(f"{D}/name_shape.json", "w"), indent=1); print("wrote name_shape.json", flush=True)
