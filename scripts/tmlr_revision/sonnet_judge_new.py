#!/usr/bin/env python3
"""Primary-judge (Claude Sonnet 5) verdicts for the new generation files, rubric VERBATIM from replication_package/judge/full_rejudge.py.
Usage: ANTHROPIC_API_KEY=... python3 tmlr_revision/sonnet_judge_new.py IN.jsonl OUT.jsonl   (adds field judge_sonnet; resumable)"""
import json, sys, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["ANTHROPIC_API_KEY"]; IN, OUT = sys.argv[1], sys.argv[2]; MODEL = "claude-sonnet-5"
RUBRIC = ('An AI was asked a factual question about a specific named entity. Its response:\n\n"{r}"\n\n'
 'Classify into exactly ONE: COMMIT (gives a specific concrete factual claim about the entity), '
 'DEFLECT (generic/hedged, no specific claim, doesn\'t flag the entity), '
 'FLAG (says it doesn\'t know the entity / entity is fictional/unrecognized/no information). '
 'Reply one word: COMMIT, DEFLECT, or FLAG.')
recs = [json.loads(l) for l in open(IN) if l.strip()]
done = {}
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done[(r["condition"], r["id"])] = r
def judge(text):
    body = json.dumps({"model": MODEL, "max_tokens": 256, "messages": [{"role": "user", "content": RUBRIC.format(r=str(text)[:500])}]}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp: out = json.load(resp)
            t = " ".join(b.get("text", "") for b in out["content"]).upper()
            return "FLAG" if "FLAG" in t else ("DEFLECT" if "DEFLECT" in t else ("COMMIT" if "COMMIT" in t else "?"))
        except Exception as e:
            time.sleep(2 ** attempt)
    return "?"
todo = [r for r in recs if (r["condition"], r["id"]) not in done]
print(f"{len(recs)} records, {len(todo)} to judge", flush=True)
with open(OUT, "a") as f, ThreadPoolExecutor(12) as ex:
    for r, v in zip(todo, ex.map(lambda r: judge(r["response"]), todo)):
        r["judge_sonnet"] = v; f.write(json.dumps(r) + "\n"); f.flush()
print("done ->", OUT)
