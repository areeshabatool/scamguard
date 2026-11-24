"""Full evaluation run, invoked by `make eval`.

The single command that reproduces every evaluation table: it runs E1-E4 in order,
each writing its own report to results/ (and per-case rows to results/raw/, git-ignored).

    python -m eval.run_eval              # live: needs ANTHROPIC_API_KEY and data/private/
    python -m eval.run_eval --dry-run    # offline: public data/samples/ + FakeLLM

Live runs read the private datasets named in config (E2_DATASET, E3_DATASET, E4_DATASET);
--dry-run uses the committed samples in data/samples/, so anyone can reproduce the tables
end to end without an API key or the private data.
"""

from __future__ import annotations

import argparse

from eval import run_e1, run_e2, run_e3, run_e4

EXPERIMENTS = (
    ("E1", run_e1.main),
    ("E2", run_e2.main),
    ("E3", run_e3.main),
    ("E4", run_e4.main),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_eval", description="Run E1-E4.")
    parser.add_argument("--dry-run", action="store_true", help="offline: samples + FakeLLM")
    args = parser.parse_args(argv)

    passthrough = ["--dry-run"] if args.dry_run else []
    mode = "dry-run" if args.dry_run else "live"
    print(f"=== Scam Guard evaluation ({mode}) ===")

    failures: list[str] = []
    for name, run in EXPERIMENTS:
        print(f"\n--- {name} ---")
        try:
            code = run(passthrough)
            if code != 0:
                failures.append(f"{name} (exit {code})")
        except Exception as exc:  # keep going so every table that can run, runs
            failures.append(f"{name} ({type(exc).__name__}: {exc})")
            print(f"  {name} FAILED: {exc}")

    print("\n=== summary ===")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    print("all evaluation tables written to results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
