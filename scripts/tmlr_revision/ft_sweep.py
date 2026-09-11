#!/usr/bin/env python3
"""TMLR revision C5: tuning sweep for the output-supervised fine-tuning arms (R-Tuning-style confidence pseudo-labels;
semantic-entropy pseudo-labels) and the probe-supervised arm. Same recipe as train_lora.py (LoRA r=16, alpha=32, dropout 0.05,
all projection modules, bs 2 x ga 8, cosine, warmup 0.05) with learning rate and epochs varied; for the SE arm the
pseudo-label threshold is varied too (abstain if SE >= train quantile q). Each config: train, then greedy 64-token generation
on the entity-disjoint lora_test. Writes ft_sweep_gens.jsonl (condition, id, label, entity, response) + ft_sweep_meta.json."""
import json, os, re, gc, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
P = "$HN_ROOT"; D = f"{P}/data/abstention_full"
MODEL = "$MODEL_ROOT/Qwen-Qwen3-8B"
MAXLEN = 320; MAXNEW = 64
GENS = f"{D}/ft_sweep_gens.jsonl"; META = f"{D}/ft_sweep_meta.json"
load = lambda f: [json.loads(l) for l in open(f)]
test = load(f"{D}/lora_test.jsonl"); train_base = load(f"{D}/lora_train.jsonl")
rt = load(f"{D}/rtuning_train.jsonl"); pt = load(f"{D}/probetuning_train.jsonl")
se_scores = {r["id"]: r["se"] for r in load(f"{D}/se_scores.jsonl")}; se_gens = {r["id"]: r for r in load(f"{D}/se_gens.jsonl")}
ABST = "I don't have reliable information to answer this confidently, so I can't provide a specific answer."
def se_rows(q):
    ses = np.array([se_scores[r["id"]] for r in train_base]); thr = float(np.quantile(ses, q))
    rows = []
    for r in train_base:
        ab = se_scores[r["id"]] >= thr
        rows.append({**{k: r[k] for k in ("id", "label", "question")}, "entity": r.get("entity"), "target": ABST if ab else se_gens[r["id"]]["greedy"]})
    fa = np.mean([(se_scores[r["id"]] >= thr) for r in train_base if r["label"] == "answerable"])
    ff = np.mean([(se_scores[r["id"]] >= thr) for r in train_base if r["label"] == "should_abstain"])
    return rows, {"thr": thr, "pseudo_abstain_rate_real": float(fa), "pseudo_abstain_rate_fic": float(ff)}
CONFIGS = [("rtuning", 5e-5, 1, 0.5), ("se", 5e-5, 1, 0.5), ("rtuning", 2e-4, 3, 0.5), ("se", 2e-4, 3, 0.5), ("se", 5e-5, 1, 0.7), ("se", 2e-4, 3, 0.7), ("se", 1e-4, 3, 0.85), ("rtuning", 1e-4, 1, 0.5)]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token
class DS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        pr = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pid = tok(pr, add_special_tokens=False)["input_ids"]; tid = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (pid + tid)[:MAXLEN]; lab = ([-100] * len(pid) + tid)[:MAXLEN]; pad = MAXLEN - len(ids)
        return {"input_ids": ids + [tok.pad_token_id] * pad, "attention_mask": [1] * len(ids) + [0] * pad, "labels": lab + [-100] * pad}
def collate(b): return {k: torch.tensor([x[k] for x in b]) for k in b[0]}
FLAG = re.compile(r"don'?t have (reliable |the )?information|not familiar|does not appear|not a (recognized|widely|known)|no reliable information|cannot provide accurate|i (don'?t|do not) (know|have)|not aware of|fictional|couldn'?t find", re.I)
done = set()
if os.path.exists(GENS): done = {json.loads(l)["condition"] for l in open(GENS)}
meta = json.load(open(META)) if os.path.exists(META) else {}
prompts = [tok.apply_chat_template([{"role": "user", "content": t["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in test]
for arm, lr, ep, q in CONFIGS:
    cond = f"{arm}_lr{lr:g}_ep{ep}" + (f"_q{q}" if q is not None and arm == "se" else "")
    if cond in done: print("skip", cond, flush=True); continue
    if arm == "rtuning": rows, info = rt, {}
    elif arm == "probetuning": rows, info = pt, {}
    else: rows, info = se_rows(q)
    print(f"=== {cond}: n_train={len(rows)} {info}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    tok.padding_side = "right"
    args = TrainingArguments(output_dir=f"{P}/results/ft_sweep_tmp", per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=ep,
                             learning_rate=lr, bf16=True, logging_steps=20, save_strategy="no", report_to=[], warmup_ratio=0.05, lr_scheduler_type="cosine", seed=0)
    tr = Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=collate); tr.train()
    model.eval(); model.config.use_cache = True; tok.padding_side = "left"
    outs = []
    for i in range(0, len(test), 16):
        inp = tok(prompts[i:i+16], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad(): o = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in o[:, inp.input_ids.shape[1]:]: outs.append(re.sub(r"<think>.*?</think>", "", tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip())
    with open(GENS, "a") as f:
        for t, x in zip(test, outs): f.write(json.dumps({"condition": cond, "id": t["id"], "label": t["label"], "entity": t.get("entity", ""), "response": x}) + "\n")
    fic = [x for t, x in zip(test, outs) if t["label"] == "should_abstain"]; real = [x for t, x in zip(test, outs) if t["label"] == "answerable"]
    rr = {"arm": arm, "lr": lr, "epochs": ep, "se_quantile": q, **info, "regex_flag_fic": float(np.mean([bool(FLAG.search(x)) for x in fic])), "regex_flag_real": float(np.mean([bool(FLAG.search(x)) for x in real])),
          "train_loss_final": float(tr.state.log_history[-1].get("train_loss", float("nan"))) if tr.state.log_history else None}
    meta[cond] = rr; json.dump(meta, open(META, "w"), indent=1); print("RESULT", cond, json.dumps(rr), flush=True)
    del model, tr; gc.collect(); torch.cuda.empty_cache()
print("FT SWEEP DONE", flush=True)
