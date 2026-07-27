#!/usr/bin/env python3
"""Hardening pass (N90): classify every parametric ANSWERABLE question as PUBLIC (a diligent
person could verify the answer from public sources) vs NONPUBLIC (internal/proprietary specifics
no public source documents). Qwen3-32B-AWQ, tp=2, greedy, thinking off. Output feeds the
verifiability-split robustness analysis; validated against the human audit sheet."""
import os, json, re
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

P = os.environ.get("HN_ROOT", ".")
JUDGE = os.environ.get("JUDGE_MODEL_DIR", "Qwen/Qwen3-32B-AWQ")

items = [json.loads(l) for l in open(f"{P}/data/items.jsonl")
         if json.loads(l)["arm"] == "parametric"]
ans = [it for it in items if it["label"] == "answerable"]
print(f"classifying {len(ans)} parametric answerable questions", flush=True)

PROMPT = """You are auditing a question dataset. Classify whether the ANSWER to the question below is publicly verifiable.

PUBLIC = a diligent researcher could find and verify the answer from public sources (company reports, standards documents, official websites, reputable press).
NONPUBLIC = the question asks for internal, proprietary, or operational specifics that organizations do not publish (e.g., a company's internal safety-stock thresholds, private vendor terms, unpublished process parameters), so no public source could verify any answer.

Question: {q}

Reply with exactly one word: PUBLIC or NONPUBLIC."""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(JUDGE, trust_remote_code=True)
def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)

llm = LLM(model=JUDGE, tensor_parallel_size=2, max_model_len=4096, enforce_eager=True,
          gpu_memory_utilization=0.9)
sp = SamplingParams(temperature=0.0, max_tokens=2048)
outs = llm.generate([chat(it["question"]) for it in ans], sp)

def strip_thinking(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

res = []
for it, o in zip(ans, outs):
    t = strip_thinking(o.outputs[0].text).upper()
    v = "NONPUBLIC" if "NONPUBLIC" in t else ("PUBLIC" if "PUBLIC" in t else "UNPARSED")
    res.append({"id": it["id"], "pair": it.get("pair") or it["id"].rsplit("_", 1)[0],
                "question": it["question"], "verifiability": v})
from collections import Counter
print(Counter(r["verifiability"] for r in res), flush=True)
with open(f"{P}/data/answerable_verifiability.jsonl", "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")
print("wrote answerable_verifiability.jsonl", flush=True)
