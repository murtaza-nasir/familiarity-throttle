# Replication Package: "Decodable but Not Actionable"

Replication materials for the companion mechanism paper (arXiv posting) and the
applied manuscript sharing the same testbed. The papers study the
represent–express gap in large language models: a linear probe on mid-layer
residual activations recovers a familiarity signal that the model's own
behavior under-uses; the mechanism paper characterizes what the signal encodes
(a graded exposure ladder), its generalization, the throttle circuit, and
conditional release.

Coverage note: `data/` and `results/` reproduce the ladder (Section 4), the
cross-domain transfer (Section 5), and the intervention comparison (Section 7)
end to end, including all judge verdicts from both judge families.
`results/mechanism/` ships the Section-6 result files (layer-resolved causal
trace, component screen and verification, circuit dependency/clamping,
dose–response summary and verdicts, optimized-steer metadata) and the
independent RAGTruth grounding-transfer measurement
(`independent_citation.json`). Raw activation caches for the mechanism runs
are too large to ship and remain on the compute pipeline; the generating
scripts are included and referenced per result. The cross-construct
(familiarity↔groundedness) transfer measurement ships as script only.

## Directory map

```
replication_package/
  data/       testbed items, leak/collision censuses, verification files,
              gate-probe training split, human audit sheet
  judge/      Sonnet judge driver, verdicts, calibration set, original
              open-weights rubric script (rubric provenance), and gens/
              with the raw judged generations
  results/    aggregated result JSONs and per-item generation/judgment files
              backing the main-text tables and figures
  scripts/    analysis and experiment scripts (anonymized, env-var paths)
```

### data/
- `items.jsonl` — the 2,042-item core testbed: 770 parametric items (385
  matched real/fictional entity pairs) plus 1,272 grounded items
  (supply-chain, marketing, and decision-support contexts). Fields include
  `id`, `arm`, `label` (answerable vs fictional/unanswerable), `entity`,
  `question`. The plausible (realistic-name), temporal, cross-domain, and
  numeric arms are separate files, listed below.
- `plausible_fictional_items.jsonl` — the realistic-name (plausible) arm
  (400 items): 200 answerable (real-entity) items and 200 fictional items
  with plausible, natural-sounding names; the morphology control for Table 1.
- `numeric_param_items.jsonl`, `numeric_pilot_items.jsonl` — the numeric arm
  items (used by the agentic-loop scripts).
- `temporal_all_items.jsonl` (+ `temporal_postcutoff_items.jsonl`,
  `temporal_fictional_items.jsonl`, `temporal_wellknown_items.jsonl`) — the
  temporal tier (213 items): post-training-cutoff entities, matched fictional,
  and well-known controls.
- `crossdomain_{academic,biomed,people}_items.jsonl`, `crossdomain_all_items.jsonl`
  — cross-domain arms (884 items total) for the domain-generalization analysis.
- `crossdomain_leak_screen_census.json`, `crossdomain_leak_verdicts_census.jsonl`,
  `census_collisions_by_domain.json`, `crossdomain_collision_ids.json` —
  census-based leak screen for the cross-domain arms (fictional names checked
  for collisions with real entities; collided ids excluded).
- `collision_census_raw.jsonl`, `plausible_collision_sensitivity.json` —
  the same census applied to the realistic-name arm, with a sensitivity
  analysis excluding potential collisions.
- `postcutoff_verification_census.json` — verification that post-cutoff
  entities are genuinely post-cutoff.
- `answerable_verifiability.jsonl` — verifiability classification of the
  answerable items (grounded-arm construction).
- `steer_v.pt` — the steering vector (torch tensor) used by the steering
  scripts; `steer_v2_meta.json` — steering-vector metadata (layer, alpha,
  norm, gate threshold).
- `lora_train.jsonl` / `lora_test.jsonl` — the entity-disjoint
  train/held-out split used for the gate probe and all remediation
  training/evaluation.
- `repr/probe_layers_summary.json` — per-model best (position, layer) from
  the layer sweep; read by `bootstrap_cis.py`. The activation caches
  themselves (`repr/*.npz`) are not shipped (large); regenerate with
  `extract_repr.py`.
- `human_audit_sheet.csv` — 98-row stratified human audit of item labels
  (columns: entity real? label correct? notes), completed by hand.

### judge/
- `full_rejudge.py` — the Claude Sonnet 5 judge driver. Reads the API key from
  the `ANTHROPIC_API_KEY` environment variable and judged-set files from
  `HN_JUDGE_DIR` (default `.`). Rubric is verbatim from
  `generic_3way_judge.py`.
- `sonnet_final.json` — all 10,691 Sonnet verdicts over every judged
  generation set in the paper (`{uid, set, key, label, verdict, model}`).
- `sonnet_calibration.jsonl` — 681-item calibration set comparing the Sonnet
  judge to the open-weights judge and human audit.
- `generic_3way_judge.py` — the original open-weights (Qwen3-32B-AWQ) judge
  script; included for rubric provenance. All headline numbers in the paper
  use the Sonnet verdicts.
- `gens/` — the raw generation files that were judged (the judge's inputs,
  with the model responses the verdicts refer to): `qwen3.jsonl`,
  `qwen3_base.jsonl`, `gemma3.jsonl`, `gemma27.jsonl`, `qwen3_32b.jsonl`,
  `qwen3_6_27b.jsonl`, `aya.jsonl`, `llama31.jsonl` (cross-model sets
  `cm_*`); `steer_v2_gens.jsonl`, `steer_llama_gens.jsonl` (steering sets
  `steer_*` / `llsteer_*`); `thinking_merged.jsonl` (`thinking` set);
  `temporal_responses.jsonl` (`temporal_*` sets). The CAST staged
  generations (`cast_stage_judged.jsonl` and the grounded equivalent) and
  `se_ft_gens.jsonl` live under `results/`.

Judge protocol: Claude Sonnet 5 (pinned `claude-sonnet-5`), three-way
COMMIT / DEFLECT / FLAG classification of the first 500 characters of each
response, `max_tokens=256`, no temperature parameter (API default), one-word
reply parsed from the message text.

### results/
- `cascade_scalars.json` — scalar quantities for the cascade figure,
  recomputed from shipped files under the Sonnet judge (provenance in the
  file's `_sources` field): probe AUROC from `bootstrap_cis.json`,
  verbalized AUROC + parse rate from `verbalized_confidence_records.json`,
  behavior rates from `judge/sonnet_final.json` set `cm_qwen3`.
- `verbalized_confidence_records.json` — per-item verbalized-confidence
  elicitation on the 770 parametric items (stated 0-100 confidence +
  response); the source of the 0.719 verbalized AUROC and Table 2's
  Verbalized column for the subject model.
- `gemma27_verbalized_records.json`, `scale30b_verbalized_records.json`,
  `qwen3_32b_verbalized_records.json`, `qwen3_6_27b_verbalized_records.json`
  — per-item verbalized-confidence records for the larger-model scale runs
  (see `scale_family.json` for aggregates).
- `scale_family.json` — aggregate scale-table results (probe, verbalized,
  behavior) across the model family, with the 8B reference row.
- `scale30b.json`, `qwen3_32b_scale_probe.json`,
  `qwen3_6_27b_scale_probe.json` — per-model probe results for the >8B
  scale runs (Gemma-3-27B, Qwen3-32B, Qwen3.6-27B).
- `se_gens.jsonl`, `se_scores.jsonl` — semantic-entropy sampled generations
  and per-item entropy scores (the semantic-entropy evidence).
- `entity_ppl.json` — entity-perplexity control (perplexity-only AUROC).
- `crossmodel_replication_census.json` — cross-model replication (probe AUROC
  and behavior across model families), census-screened items.
- `crossdomain_transfer.json` — cross-domain probe transfer matrix
  (census-screened version; produced by `crossdomain_transfer_census.py`).
- `bootstrap_cis.json` — group-bootstrap confidence intervals for headline numbers.
- `probe_predicts_behavior_sonnet.json` — probe-score vs. behavior analysis
  under Sonnet verdicts.
- `policy_economics_sonnet.json` — disposition-policy economics (cost-optimal
  policy regions) under Sonnet verdicts.
- CAST remediation: `cast_stage_items.jsonl`, `cast_stage_judged.jsonl`,
  `cast_stage_meta.json` (staged generations and judgments, parametric) and
  the grounded equivalents `castg_stage_items.jsonl`,
  `castg_stage_judged.jsonl`, `castg_stage_meta.json`;
  `crossdomain_cast_gens.jsonl` + `crossdomain_cast_verdicts.json`,
  `plausible_cast_gens.jsonl` + `plausible_cast_verdicts.json`.
- Remediation baselines: generations in `se_ft_gens.jsonl` (semantic-entropy
  fine-tune) and `judge/gens/steer_v2_gens.jsonl` /
  `judge/gens/steer_llama_gens.jsonl` (steering); verdicts for all baselines
  (LoRA, R-Tuning, probe-tuning, semantic-entropy FT, steering variants) are
  the `steer_*`, `llsteer_*`, and `seft_*` sets in `judge/sonnet_final.json`.
- Live agent: `agentic_loop_cast_gens.jsonl` (per-episode generations),
  `agentic_loop_cast_sonnet.json` (aggregates under Sonnet verdicts),
  `agentic_loop_heldout.json` (held-out gate threshold selection).

### scripts/
All scripts are anonymized: absolute paths were replaced with environment
variables, keeping the logic byte-for-byte identical otherwise.

- `HN_ROOT` (default `.`) — package root; scripts read/write `$HN_ROOT/data/...`.
- `HN_DATA` (default `./data`) — data directory, where used directly.
- `MODEL_DIR` (default `Qwen/Qwen3-8B`) — subject model (HF id or local path).
- `JUDGE_MODEL_DIR` (default `Qwen/Qwen3-32B-AWQ`) — open-weights judge.
- `ANTHROPIC_API_KEY` — required by the Sonnet judge driver only.

Run scripts from the package root (`cd replication_package && python scripts/...`)
so the default relative paths resolve. Some scripts expect cached activation
files under `data/repr/` (produced by `extract_repr.py`); regenerate these
locally from the item files — activations are deterministic given the public
weights (bf16).

Key scripts:
- `extract_repr.py` — extracts mean-pooled residual representations (the probe
  input) from the subject model.
- `bootstrap_cis.py` — probe fitting and group-bootstrap CIs.
- `entity_perplexity.py` — entity-perplexity control.
- `classify_verifiable.py` — verifiability classification of answerables.
- `crossdomain_transfer_census.py`, `crossmodel_replication_census.py` —
  domain and model generalization (census-screened).
- `crossdomain_cast.py`, `plausible_cast.py` — CAST remediation on the
  cross-domain and realistic-name arms.
- `steer_v2.py`, `steer_optimize.py` — steering-vector remediation.
- `train_lora.py`, `eval_lora.py`, `prep_rtuning.py`, `prep_se_tuning.py`,
  `eval_se_ft.py`, `prep_probetuning.py` — remediation baselines
  (LoRA, R-Tuning, semantic-entropy FT, probe-tuning).
- `agentic_loop_cast.py`, `agentic_loop_extract.py`, `agentic_loop_eval.py` —
  live-agent cascade (generation, activation caching, evaluation).
- `policy_economics_sonnet.py` — disposition-policy economics and policy-region
  figure.
- `probe_predicts_behavior_sonnet.py` — per-item probe-score vs. behavior.
- `longcontext_gate.py`, `labelfree_v2.py` — long-context gate check and
  label-free gate variant.

## Probe protocol (used throughout)

Mean-pooled residual-stream representations at a mid layer of the subject
model; StandardScaler followed by L2-regularized logistic regression (C=0.3);
GroupKFold with 5 folds grouped by matched pair (pair-disjoint folds);
out-of-fold AUROC; confidence intervals by group bootstrap (resampling pairs).
Activations are extracted in bf16.

## Table / figure map

| Paper element | Files |
|---|---|
| Table 1 (morphology control: realistic-name arm) | `data/plausible_collision_sensitivity.json`, `data/plausible_fictional_items.jsonl`, `data/collision_census_raw.jsonl`, `results/plausible_cast_gens.jsonl`, `results/plausible_cast_verdicts.json` |
| Table 2 (scale) | behavior: `judge/sonnet_final.json` sets `cm_*` (raw generations in `judge/gens/`); 8B-class probes: `results/bootstrap_cis.json`; >8B probes: `results/scale30b.json` (Gemma-3-27B), `results/qwen3_32b_scale_probe.json`, `results/qwen3_6_27b_scale_probe.json`, aggregates in `results/scale_family.json`; Verbalized column: `results/verbalized_confidence_records.json` + the per-model `*_verbalized_records.json` files |
| Table 3 (remedies) | `judge/sonnet_final.json` sets `steer_*` / `llsteer_*` / `seft_*`; generations: `results/se_ft_gens.jsonl`, `judge/gens/steer_v2_gens.jsonl`, `judge/gens/steer_llama_gens.jsonl`; CAST stages: `results/cast_stage_*` and `results/castg_stage_*`; `data/steer_v.pt`, `data/steer_v2_meta.json` |
| Table 4 (live agent) | `results/agentic_loop_cast_sonnet.json`, `results/cast_stage_*` / `results/castg_stage_*`, `results/agentic_loop_cast_gens.jsonl`, `results/agentic_loop_heldout.json` |
| Fig. cascade | scalars: `results/cascade_scalars.json` (regenerated from `judge/sonnet_final.json`, `results/bootstrap_cis.json`, `results/verbalized_confidence_records.json`); `results/agentic_loop_cast_sonnet.json` |
| Fig. domain generalization | `results/crossdomain_transfer.json`, `results/crossdomain_cast_gens.jsonl`, `results/crossdomain_cast_verdicts.json`, `data/crossdomain_*_items.jsonl` |
| Fig. policy regions | `results/policy_economics_sonnet.json` (produced by `scripts/policy_economics_sonnet.py`) |

All judged numbers in the tables derive from `judge/sonnet_final.json`;
`judge/sonnet_calibration.jsonl` and `data/human_audit_sheet.csv` back the
judge-validation paragraph.

## Supersession note

Some per-item files carry embedded verdict fields from an earlier
(open-weights) judge run: `results/cast_stage_judged.jsonl`,
`results/castg_stage_judged.jsonl`, and `results/agentic_loop_cast_gens.jsonl`.
Those embedded verdicts are superseded by `judge/sonnet_final.json` (sets
`aparam_*` / `agrnd_*` for the agentic and staged runs). All numbers in the
paper use the Sonnet verdicts in `sonnet_final.json`; the embedded fields are
retained only because they are part of the original generation records.

## Data notes

- `judge/sonnet_final.json` also contains a `thinking` set (578 rows) that is
  not used in the paper.
- In `judge/sonnet_final.json`, the `aparam_normal` set contains 216 grounded
  rows duplicated with `agrnd_normal` (an artifact of a merged input file);
  the verdicts agree on all 216 rows.
- In `results/policy_economics_sonnet.json`, `escalated_unsup: 73` appears
  beside the none-arm's `n_unsup: 72`. Denominators are per-policy: the
  block's own `n` is 73, so its escalation rate is 73/73 = 1.0.

## Files not shipped

- `data/repr/*.npz` activation caches — large; regenerate with
  `scripts/extract_repr.py` (deterministic given the public weights, bf16).
- Long-context gate results — not shipped (the run prints its AUROC /
  gate-fire table to stdout); regenerable via `scripts/longcontext_gate.py`.
- Label-free gate-variant results — not shipped (stdout only); regenerable
  via `scripts/labelfree_v2.py` (reads
  `data/postcutoff_verification_census.json` and `results/entity_ppl.json`).

## Environment

Python 3.10+. Required packages: `torch`, `transformers`, `scikit-learn`,
`numpy`, `anthropic` (or plain `urllib`, as in `full_rejudge.py`) for the
judge, `matplotlib` for figures, `vllm` (optional) for fast generation,
`peft` for the LoRA baselines.

Paths: scripts default to `HN_DATA=./data` (and `HN_ROOT=.`), but several
input files they read are shipped under `results/`. Set `HN_DATA` to a
directory containing both the `data/` and `results/` contents, or symlink
the needed `results/` files into `data/`, before running.
`judge/full_rejudge.py` writes `sonnet_verdicts.jsonl`; that raw per-call
output was consolidated into the shipped `judge/sonnet_final.json`.

Hardware: subject-model experiments (probe extraction, steering, generation;
Qwen3-8B and Llama-3.1-8B in bf16) run on a single 24 GB GPU. The larger
models (Gemma-3-27B, Qwen3-32B, Qwen3.6-27B) need roughly 2x24 GB or one
higher-memory GPU. Activations must be computed in bf16 — FP8 quantization
distorts activation magnitudes and invalidates the probe. The behavioral
judge is API-based (Anthropic, pinned `claude-sonnet-5`; the full ~10,700-call
re-judge cost on the order of $10), so no local hardware is needed for
judging. Analysis scripts are CPU-only.

Model weights are public on HuggingFace:
- Subject: `Qwen/Qwen3-8B` (plus `Qwen/Qwen3-0.6B/1.7B/4B/14B/32B`,
  `Qwen/Qwen3-30B-A3B` for scale analyses)
- Cross-model replication: `meta-llama/Llama-3.1-8B-Instruct`,
  `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
- Open-weights judge (provenance only): `Qwen/Qwen3-32B-AWQ`

Generation and activation extraction use bf16; probe results are deterministic
given fixed seeds (set in the scripts).


## License and citation

Code is MIT-licensed (see LICENSE). Dataset files are released for research
use; please cite the paper:

> Murtaza Nasir. Decodable but Not Actionable: Localizing and Releasing the
> Familiarity Throttle in Language Models. arXiv preprint, 2026.
> (arXiv ID to be added on announcement.)


## Scope note

`judge/` verdict files span the research program's full judged corpus,
including evaluation arms from earlier testbed designs that the paper does not
use; the corpus-wide judge-triangulation numbers recompute from the verdict
files alone. Dataset files in `data/` are the paper's final instruments; the
candidates and exclusion records are retained as construction provenance.
