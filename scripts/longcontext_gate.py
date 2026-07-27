#!/usr/bin/env python3
"""C: familiarity gate under long-context recall (fills the disclosed gap).
Embeds each parametric question in a ~750-token operations-agent preamble and measures
L18 probe AUROC + gate fire rates under (i) naive full-prompt mean-pooling and
(ii) question-span-only pooling. Probe/gate = lora_train-trained, thresholds from steer_v2_meta."""
import os
import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
P=os.environ.get("HN_ROOT", "."); D=f"{P}/data"
MODEL=os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
L=18
meta=json.load(open(f"{D}/steer_v2_meta.json")); GATE_TH=float(meta["cast_threshold"])
PREAMBLE=("You are the recall module of an enterprise supply-chain operations agent. The pipeline you serve "
"handles procurement, logistics planning, supplier onboarding, and demand forecasting for a multinational "
"manufacturer. Upstream modules have already parsed the user request, routed it to you, and determined that "
"the answer should come from your internal knowledge rather than from document retrieval. Your answer will be "
"passed to a downstream computation module that extracts any numeric quantity you state and uses it in a "
"purchase-order sizing calculation, so precision matters and fabricated values are costly. Company policy "
"requires that recalled facts about suppliers, standards, part numbers, lead times, and contractual terms be "
"accurate as of your training data; if a fact is outside your knowledge you should say so plainly. The current "
"session covers the EMEA region, fiscal quarter Q3, with standard incoterms unless stated otherwise. Earlier "
"in this session the user asked about carrier surcharge schedules, pallet dimension standards, and two supplier "
"qualification questions, all of which were answered from internal knowledge and verified downstream without "
"incident. Treat the following question in the same way: answer concisely, state numbers plainly, and do not "
"speculate beyond what you know. "*3)
items=[json.loads(l) for l in open(f"{D}/items.jsonl") if json.loads(l)["arm"]=="parametric"]
train=[json.loads(l) for l in open(f"{D}/lora_train.jsonl")]
tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,output_hidden_states=True,trust_remote_code=True).to("cuda:0").eval()
print("preamble tokens:", len(tok(PREAMBLE)["input_ids"]), flush=True)
def reps(rows, preamble):
    full=[]; qspan=[]
    for i in range(0,len(rows),4):
        batch=rows[i:i+4]
        prompts=[tok.apply_chat_template([{"role":"user","content":(preamble+r["question"]) if preamble else r["question"]}],
                 tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
        qlens=[len(tok(r["question"],add_special_tokens=False)["input_ids"]) for r in batch]
        inp=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=2048).to("cuda:0")
        with torch.no_grad(): out=model(**inp)
        h=out.hidden_states[L]; m=inp.attention_mask
        for b in range(len(batch)):
            valid=m[b].bool()
            hv=h[b][valid]
            full.append(hv.mean(0).float().cpu().numpy())
            # question span: last (qlen + ~5 template tokens) positions before generation prompt; approximate:
            k=min(qlens[b]+8, hv.shape[0])
            qspan.append(hv[-k:].mean(0).float().cpu().numpy())
        del out
    return np.array(full), np.array(qspan)
# gate trained exactly as CAST (bare short questions, full-pool == qspan for short prompts)
Xtr,_=reps(train,None)
ytr=np.array([1 if r["label"]=="should_abstain" else 0 for r in train])
sc=StandardScaler().fit(Xtr); clf=LogisticRegression(C=0.3,max_iter=2000).fit(sc.transform(Xtr),ytr)
y=np.array([1 if it["label"]=="should_abstain" else 0 for it in items])
for tag,pre in [("short",None),("longctx",PREAMBLE)]:
    F,Q=reps(items,pre)
    for pool,X in [("fullpool",F),("qspan",Q)]:
        s=clf.predict_proba(sc.transform(X))[:,1]
        fire_f=(s[y==1]>GATE_TH).mean(); fire_r=(s[y==0]>GATE_TH).mean()
        print(f"{tag:8s} {pool:8s} AUROC={roc_auc_score(y,s):.4f}  gate_fire fic={fire_f:.3f} real={fire_r:.3f}", flush=True)
print("DONE_LONGCTX", flush=True)
