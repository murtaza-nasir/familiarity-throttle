#!/usr/bin/env python3
"""Generic all-layer / two-position representation extractor for the abstention set.
Captures, per item: mean-pooled prompt hidden state AND last-prompt-token hidden state at
EVERY layer. Serves: layer/token-resolved probe (T1), base-vs-instruct (T2), cross-model (T2).
Architecture-agnostic via output_hidden_states. One model per GPU (set CUDA_VISIBLE_DEVICES).

  python extract_repr.py --model_path ... --out_tag qwen3_instruct --instruct
  python extract_repr.py --model_path ...-Base --out_tag qwen3_base   --base
"""
import os, json, argparse
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("HN_ROOT", ".")
OUT = f"{P}/data/repr"
os.makedirs(OUT, exist_ok=True)


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model_path", required=True)
    a.add_argument("--out_tag", required=True)
    a.add_argument("--instruct", action="store_true")
    a.add_argument("--base", action="store_true")
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--max_len", type=int, default=1536)
    a.add_argument("--multi_gpu", action="store_true")
    a.add_argument("--input_file", default=f"{P}/data/items.jsonl")
    return a.parse_args()


def main():
    args = parse()
    items = [json.loads(l) for l in open(args.input_file)]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    mk = dict(torch_dtype=torch.bfloat16, output_hidden_states=True, trust_remote_code=True)
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", **mk).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **mk).to("cuda:0").eval()
    IN_DEV = next(model.parameters()).device

    def build(it):
        if it["arm"] == "grounded":
            uc = ("Read the following passage and answer the question based ONLY on the information "
                  "provided.\n\nPassage: " + it["passage"][:3000] + "\n\nQuestion: " + it["question"])
        else:
            uc = it["question"]
        if args.instruct:
            try:
                return tok.apply_chat_template([{"role": "user", "content": uc}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template([{"role": "user", "content": uc}], tokenize=False,
                                               add_generation_prompt=True)
        return uc + "\nAnswer:"

    prompts = [build(it) for it in items]
    mp, lt = None, None
    for i in range(0, len(items), args.batch):
        inp = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(IN_DEV)
        with torch.no_grad():
            out = model(**inp)
        nL = len(out.hidden_states)
        if mp is None:
            H = out.hidden_states[0].shape[-1]
            mp = [[] for _ in range(nL)]; lt = [[] for _ in range(nL)]
        for L in range(nL):
            h = out.hidden_states[L]
            m = inp.attention_mask.unsqueeze(-1).to(h.device)
            mp[L].append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy().astype(np.float16))
            lt[L].append(h[:, -1, :].float().cpu().numpy().astype(np.float16))  # left-padded => last real token
        del out
        if (i // args.batch) % 40 == 0:
            print(f"  {args.out_tag}: {i+args.batch}/{len(items)}", flush=True)

    meanpool = np.stack([np.concatenate(x, 0) for x in mp])  # (L, N, H)
    lasttok = np.stack([np.concatenate(x, 0) for x in lt])
    np.savez(f"{OUT}/{args.out_tag}.npz", meanpool=meanpool, lasttok=lasttok,
             arm=np.array([it["arm"] for it in items]),
             dom=np.array([it.get("domain", it.get("tier", "na")) for it in items]),
             tier=np.array([it.get("tier", "na") for it in items]),
             y=np.array([1 if it["label"] == "should_abstain" else 0 for it in items]),
             grp=np.array([it["id"].rsplit("_", 1)[0] for it in items]),
             nlayers=meanpool.shape[0])
    print(f"{args.out_tag} DONE: {meanpool.shape[0]} layers, {meanpool.shape[1]} items, H={meanpool.shape[2]}", flush=True)


if __name__ == "__main__":
    main()
