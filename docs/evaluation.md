# Evaluation methodology

All evaluation data is **synthetic**. Every runner has a `--dry-run` mode (scripted
FakeLLM, public `data/samples/`) for offline reproduction, and a live mode that reads the
private synthetic datasets in `data/private/` (git-ignored) and needs `ANTHROPIC_API_KEY`.

Reproduce every table with one command:

```bash
make eval                        # live
python -m eval.run_eval --dry-run  # offline
```

## Systems compared (all behind `predict(text)`)

- **B0** keyword/regex rules (no model)
- **B1** a single plain prompt (message pasted into the prompt)
- **B2** a hardened prompt (message wrapped as untrusted data)
- **B3** the full Scam Guard pipeline (all code guards)

## Experiments

| ID | Question | Output |
| -- | -------- | ------ |
| **E1** | Detection quality: recall, FPR, abstain rate, turns, cost, latency for B0–B3 | `results/e1.json` |
| **E2** | Prompt-injection attack success rate (control vs. `base + attack`) for B1/B2/B3, per category | `results/e2.json` |
| **E3** | Does the conformal gate keep the realized miss rate within `alpha`? Random calibration/test splits, plus a held-out scam type | `results/e3.json` |
| **E4** | English vs Roman Urdu parity: paired false-positive difference and per-language recall/FPR | `results/e4.json` |

Reports contain **metrics only** (no message text). Per-case rows (ids/labels/scores) go
to `results/raw/`, which is git-ignored. Each report is tagged with the git commit and
the random seeds used, so a run is reproducible.

## Statistics

- 95% confidence intervals are percentile bootstraps with a fixed seed (`eval/metrics.py`).
- E1 reports a paired bootstrap difference (B3 vs B0 recall).
- E3 reports mean and spread (std, 2.5/97.5 percentiles, min, max) across splits.

## Calibration

The gate needs scam scores to set its threshold. Build `data/calibration.json` with:

```bash
python -m eval.calibrate --dataset <scams.jsonl>   # live
python -m eval.calibrate --dry-run                 # offline smoke
```

Without a calibration file the gate can never release benign (fail-safe), so every
otherwise-benign message is returned as `unsure`.

## Failure attribution

`python -m eval.failure_tags --dataset <gold.jsonl>` tags each miss and false alarm as
`ocr_error`, `extraction_error`, `tool_gap`, `model_misjudgement` or `gate_abstention`,
using the dataset's `gold_urls` / `gold_requests` / `gold_authority_claim` and the OCR
confidence, and writes a count table plus a review CSV.
