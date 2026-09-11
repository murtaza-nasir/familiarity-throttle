#!/usr/bin/env python3
"""TMLR revision C3: familiarity probe on an external corpus the authors did not build (PopQA, Mallen et al. 2023).
Probe: exact steer_v2/graded_tier protocol (Qwen3-8B, mean-pooled hidden_states[18], StandardScaler+LogReg C=0.3 on lora_train).
For a popularity-stratified PopQA sample: probe score vs log10 subject page views; probe as predictor of answer error;
greedy generation (64 tok) scored by alias match. Writes popqa_external.jsonl + popqa_external.json"""
import json, re, math, numpy as np, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
P = "$HN_ROOT"; D = f"{P}/data/abstention_full"
MODEL = "$MODEL_ROOT/Qwen-Qwen3-8B"
L_V = 18; MAXNEW = 64; N = int(os.environ.get("N_POPQA", "3000")); SEED = 0
from datasets import load_dataset
ds = load_dataset("akariasai/PopQA", split="test")
rows = [r for r in ds if r.get("s_pop") is not None and r["s_pop"] > 0]
rng = np.random.default_rng(SEED)
lp = np.array([math.log10(r["s_pop"]) for r in rows])
bins = np.quantile(lp, np.linspace(0, 1, 11)); idx = []
per = N // 10
for b in range(10):
    m = np.where((lp >= bins[b]) & ((lp < bins[b+1]) if b < 9 else (lp <= bins[b+1])))[0]
    idx += list(rng.choice(m, min(per, len(m)), replace=False))
items = [rows[i] for i in idx]
print(f"PopQA total {len(rows)} sampled {len(items)}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
def prompts(qs): return [tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in qs]
@torch.no_grad()
def pooled(qs):
    out = []; pr = prompts(qs)
    for i in range(0, len(qs), 8):
        inp = tok(pr[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
        hs = model(**inp, output_hidden_states=True).hidden_states[L_V]
        am = inp["attention_mask"].unsqueeze(-1).float()
        out.append(((hs.float()*am).sum(1)/am.sum(1)).cpu().numpy())
    return np.concatenate(out)
train = [json.loads(l) for l in open(f"{D}/lora_train.jsonl")]
Xtr = pooled([r["question"] for r in train]); ytr = np.array([1 if r["label"]=="should_abstain" else 0 for r in train])
sc = StandardScaler().fit(Xtr); probe = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
print("probe built", flush=True)
qs = [r["question"] for r in items]
scores = probe.predict_proba(sc.transform(pooled(qs)))[:,1]
print("scored", flush=True)
FLAG = re.compile(r"don'?t have (reliable |the )?information|not familiar|does not appear|not a (recognized|widely|known)|no reliable information|cannot provide accurate|i (don'?t|do not) (know|have)|not aware of|fictional|couldn'?t find|no information|unable to find|not sure", re.I)
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()
recs = []; pr = prompts(qs)
for i in range(0, len(items), 16):
    inp = tok(pr[i:i+16], return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad(): o = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
    for it, s, row in zip(items[i:i+16], scores[i:i+16], o[:, inp.input_ids.shape[1]:]):
        g = re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip()
        ans = json.loads(it["possible_answers"]) if isinstance(it["possible_answers"], str) else it["possible_answers"]
        gn = norm(g); correct = any(norm(a) and norm(a) in gn for a in ans)
        recs.append({"id": it["id"], "subj": it["subj"], "prop": it["prop"], "s_pop": it["s_pop"], "log_pop": math.log10(it["s_pop"]),
                     "question": it["question"], "probe_score": float(s), "gen": g, "correct": bool(correct), "flag_regex": bool(FLAG.search(g))})
    if (i//16) % 20 == 0: print(f"gen {i+len(inp.input_ids)}/{len(items)}", flush=True)
with open(f"{D}/popqa_external.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False)+"\n")
ps = np.array([r["probe_score"] for r in recs]); lpop = np.array([r["log_pop"] for r in recs]); cor = np.array([r["correct"] for r in recs]); fl = np.array([r["flag_regex"] for r in recs])
rho, p = stats.spearmanr(ps, lpop)
q = np.quantile(lpop, [1/3, 2/3]); lo = lpop <= q[0]; hi = lpop >= q[1]
auc_pop = roc_auc_score(np.r_[np.ones(lo.sum()), np.zeros(hi.sum())], np.r_[ps[lo], ps[hi]])
auc_err = roc_auc_score(~cor, ps) if 0 < (~cor).sum() < len(cor) else None
auc_flag = roc_auc_score(fl, ps) if 0 < fl.sum() < len(fl) else None
quint = np.digitize(ps, np.quantile(ps, [.2,.4,.6,.8]))
by_q = [{"probe_quintile": int(k), "n": int((quint==k).sum()), "accuracy": float(cor[quint==k].mean()), "flag_rate": float(fl[quint==k].mean()), "mean_log_pop": float(lpop[quint==k].mean())} for k in range(5)]
dec = np.digitize(lpop, np.quantile(lpop, np.linspace(0.1,0.9,9)))
by_pop = [{"pop_decile": int(k), "n": int((dec==k).sum()), "mean_probe": float(ps[dec==k].mean()), "accuracy": float(cor[dec==k].mean()), "flag_rate": float(fl[dec==k].mean())} for k in range(10)]
summ = {"n": len(recs), "spearman_probe_vs_logpop": {"rho": float(rho), "p": float(p)}, "auroc_probe_lowpop_vs_highpop_tertiles": float(auc_pop),
        "auroc_probe_predicts_incorrect": auc_err, "auroc_probe_predicts_regex_flag": auc_flag, "overall_accuracy": float(cor.mean()), "overall_flag": float(fl.mean()),
        "by_probe_quintile": by_q, "by_pop_decile": by_pop, "protocol": "Qwen3-8B; probe = lora_train mean-pooled hidden_states[18], StandardScaler+LogReg(C=0.3); PopQA test split, 10-bin popularity-stratified sample, seed 0; greedy 64 tokens; correctness = alias substring match"}
json.dump(summ, open(f"{D}/popqa_external.json","w"), indent=1)
print(json.dumps(summ, indent=1), flush=True)
