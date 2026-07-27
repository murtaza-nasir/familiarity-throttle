#!/usr/bin/env python3
"""Instructed-abstention arm on the full parametric set (770 items), for the policy
figure's abstain policy: same harness/decode budget as steer_v2.py (greedy, max_new=64,
enable_thinking=False). Instruction reconstructs the original 'say so if you lack
reliable info' condition. Writes instructed_arm_gens.jsonl for Sonnet judging."""
import json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("HN_ROOT", ".")
D = f"{P}/data"
MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-8B")
MAXNEW = 64

INSTR = ("If you are unsure or do not have reliable information about the entity in "
         "question, say so explicitly instead of guessing.")

items = [json.loads(l) for l in open(f"{D}/items.jsonl")]
par = [it for it in items if it["arm"] != "grounded"]
print(f"parametric items: {len(par)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda:0").eval()

prompts = [tok.apply_chat_template(
    [{"role": "user", "content": f"{it['question']}\n\n{INSTR}"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False) for it in par]

outs = []
for i in range(0, len(par), 16):
    inp = tok(prompts[i:i+16], return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    for row in o[:, inp.input_ids.shape[1]:]:
        outs.append(re.sub(r"<think>.*?</think>", "",
                    tok.decode(row, skip_special_tokens=True), flags=re.DOTALL).strip())
    if (i // 16) % 8 == 0: print(f"{i+16}/{len(par)}", flush=True)

with open(f"{D}/instructed_arm_gens.jsonl", "w") as f:
    for it, g in zip(par, outs):
        f.write(json.dumps({"id": it["id"], "label": it["label"],
                            "question": it["question"], "gen": g}) + "\n")
print("WROTE instructed_arm_gens.jsonl", flush=True)
