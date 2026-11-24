# Scam Guard

A defensive agent that reads a forwarded message or screenshot (English, Roman Urdu or
Urdu) and returns **scam / suspicious / benign** — or **unsure** — with a short piece of
safety advice in Roman Urdu. It is built as a 9-step pipeline where only three steps call
a model; the rest are deterministic code guards. All evaluation data is synthetic.

> This is a course project and a defensive tool. It never opens or fetches URLs, never
> stores forwarded content by default, and fails safe: any model or schema error, or an
> unreadable input, returns `unsure` — never `benign`.

## Requirements

- Python **3.11+**
- An Anthropic API key — only for *live* model calls (see [Environment variables](#environment-variables)).
  The test suite and the `--dry-run` evaluation need **no** key.

## Setup

```bash
git clone https://github.com/areeshabatool/scamguard.git
cd scamguard

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

This installs the `scamguard` package in editable mode plus the dev tools (pytest, ruff,
hypothesis).

## Environment variables

| Variable            | Required for                               | Notes                                  |
| ------------------- | ------------------------------------------ | -------------------------------------- |
| `ANTHROPIC_API_KEY` | live model calls (CLI, web `/analyze`, live eval) | Not needed for tests or `--dry-run`. |

Copy the template and fill in your key locally (never commit `.env`):

```bash
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env:  ANTHROPIC_API_KEY=sk-ant-...
```

The model name, thresholds and the conformal `alpha` live in
[`app/scamguard/config.py`](app/scamguard/config.py), not in environment variables.

## Running the tests

Everything runs offline with a scripted fake model — no API key, no network:

```bash
pytest -q
```

Lint and format:

```bash
ruff check .
ruff format .
```

> Troubleshooting: the property tests use [Hypothesis](https://hypothesis.works/), which
> loads a native library. On some locked-down setups that library may be blocked; if
> `pytest` errors while importing Hypothesis, skip those tests with
> `pytest -q --ignore=tests/test_properties.py`.

## Running the demo

Classify a single message from the command line (needs `ANTHROPIC_API_KEY`):

```bash
python -m scamguard.cli analyze "Your account is blocked. Send your OTP now to unlock it."
# add --json for the full result with the per-step trace
```

Or run the web UI and open <http://127.0.0.1:8000> to paste a message or upload a
screenshot:

```bash
uvicorn scamguard.api:app --reload
```

`GET /health` and the upload page need no key; `POST /analyze` makes a live model call.

**No API key?** Reproduce the whole system offline with the dry-run evaluation below — it
exercises the full pipeline on the sample data using a deterministic fake model.

## Reproduce all evaluation tables

One command runs every experiment (E1–E4) and writes each table to `results/`:

```bash
make eval                         # live: needs ANTHROPIC_API_KEY and the private datasets
# or, with no key and no private data (fully offline, from the committed samples):
python -m eval.run_eval --dry-run
```

(`make eval` just calls `python -m eval.run_eval`.) Outputs:

| File                      | Experiment                                                        |
| ------------------------- | ----------------------------------------------------------------- |
| `results/e1[_dryrun].json`| E1 — recall, FPR, abstain, turns, cost and latency for B0–B3       |
| `results/e2[_dryrun].json`| E2 — prompt-injection attack success rate for B1/B2/B3            |
| `results/e3[_dryrun].json`| E3 — does the conformal gate keep the miss rate within `alpha`?   |
| `results/e4[_dryrun].json`| E4 — English vs Roman Urdu parity                                 |

Each report is metrics only (no message text); per-case rows go to `results/raw/`, which
is git-ignored. A live run reads the synthetic private datasets named in
[`config.py`](app/scamguard/config.py) (`data/private/`, also git-ignored); the dry run
reads the committed samples in [`data/samples/`](data/samples/).

### Baselines (all behind `predict(text)`)

- **B0** keyword/regex rules (no model)
- **B1** a single plain prompt
- **B2** a hardened prompt (message wrapped as untrusted data)
- **B3** the full Scam Guard pipeline

### Failure attribution (optional)

Given a gold-annotated dataset (`gold_urls`, `gold_requests`, `gold_authority_claim`),
tag every miss and false alarm by cause and write a count table plus a review CSV:

```bash
python -m eval.failure_tags --dry-run --dataset <gold.jsonl>
```

## Layout

```
app/scamguard/   the package: the 9 pipeline steps, llm.py gateway, api.py, cli.py
data/            official_domains.json allowlist; samples/ (synthetic eval data)
eval/            metrics, baselines (systems.py), experiment runners (run_e1..e4, run_eval)
tests/           offline tests (FakeLLM only; never call the API)
results/         evaluation tables (results/raw/ is git-ignored)
docs/
```

## Commands reference

| Task                         | Command                                         |
| ---------------------------- | ----------------------------------------------- |
| Install                      | `pip install -r requirements.txt`               |
| Tests (offline)              | `pytest -q`  (or `make test`)                   |
| Lint / format                | `ruff check . && ruff format .`  (or `make lint`)|
| Analyze one message          | `python -m scamguard.cli analyze '...'`         |
| Serve the web UI             | `uvicorn scamguard.api:app --reload`  (or `make serve`) |
| All evaluation tables        | `make eval`  /  `python -m eval.run_eval --dry-run` |

`make` targets are in the [`Makefile`](Makefile); on Windows without `make`, use the
`python -m ...` / `pytest` / `ruff` commands directly.
