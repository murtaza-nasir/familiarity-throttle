import json, re, numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
D="$HN_ROOT/data/abstention_full"
rows=[json.loads(l) for l in open(f"{D}/popqa_external.jsonl")]
ds=load_dataset("akariasai/PopQA", split="test"); ans={r["id"]:(json.loads(r["possible_answers"]) if isinstance(r["possible_answers"],str) else r["possible_answers"]) for r in ds}
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",s.lower())).strip()
def wb(gen, aliases):
    g=" "+norm(gen)+" "; return any(len(norm(a))>0 and (" "+norm(a)+" ") in g for a in aliases)
for r in rows: r["correct_wb"]=wb(r["gen"], ans[r["id"]])
ps=np.array([r["probe_score"] for r in rows]); cor=np.array([r["correct_wb"] for r in rows]); lp=np.array([r["log_pop"] for r in rows])
q=np.digitize(ps, np.quantile(ps,[.2,.4,.6,.8])); d=np.digitize(lp, np.quantile(lp, np.linspace(0.1,0.9,9)))
out={"overall_accuracy":float(cor.mean()),"auroc_probe_predicts_incorrect":float(roc_auc_score(~cor,ps)),"acc_by_probe_quintile":[float(cor[q==k].mean()) for k in range(5)],"acc_by_pop_decile":[float(cor[d==k].mean()) for k in range(10)],"n_changed":int(sum(r["correct"]!=r["correct_wb"] for r in rows)),"note":"alias must match as whole words after normalization; the original correct flag used bare substring matching"}
print(json.dumps(out))
with open(f"{D}/popqa_external.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
s=json.load(open(f"{D}/popqa_external.json")); s["word_boundary_rescore"]=out; json.dump(s, open(f"{D}/popqa_external.json","w"), indent=1)
