"""Full evaluation run, invoked by `make eval`.

Runs the pipeline over the synthetic evaluation set, calibrates the benign threshold,
and writes summary metrics to results/ (raw per-message outputs go to results/raw/, not committed).
"""
