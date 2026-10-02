# Replication Package

This repository accompanies the paper and provides the analysis scripts, aggregated data, and figures needed to verify and reproduce the results.

## Structure

- `scripts/` — Analysis scripts that compute convergence scores, statistical tests, and generate figures from the aggregated data.
- `data/` — Aggregated analysis outputs (CSV, JSON, JSONL) that directly support all tables, figures, and numbers in the paper.
- `figures/` — The three figures included in the paper.

## Scripts

The scripts below implement the full analysis. They are listed in execution order, where each step builds on the outputs of the previous ones.

**Library modules** (shared utilities, not executed directly):

- `analysis_config.py` — Shared configuration: paths, model names, run counts, convergence buckets.
- `analysis_data_io.py` — Shared utilities for reading and writing JSON, JSONL, and CSV files.
- `analysis_similarity.py` — Syntactic and semantic patch similarity functions.
- `analysis_convergence.py` — Within-model convergence computation.
- `analysis_cross_model.py` — Cross-model agreement computation.
- `analysis_stats.py` — Descriptive statistics per convergence bucket.
- `analysis_plot.py` — Figure generation from aggregated data.

**Analysis steps** (run in order from `scripts/`):

1. `analysis_complementarity.py` — Per-run complementarity analysis and Figure 1.
2. `analysis_run.py --force` — Within-model convergence and cross-model agreement computation.
3. `analysis_cost_effectiveness.py --force` — Cost-effectiveness analysis and Figures 2 and 3.
4. `analysis_convergence_to_incorrect_patches.py` — Characterizes bugs where convergence produces incorrect patches.
5. `analysis_difficulty.py` — Convergence correctness stratified by bug difficulty (Tables 1 and 2).
6. `analysis_stats_tests.py` — Fisher's exact tests for statistical significance.
7. `analysis_multiple_testing.py` — Benjamini-Hochberg correction for multiple comparisons.
8. `analysis_sensitivity_n.py` — Sensitivity analysis varying the number of runs (N=3 to 6).

## Data

The `data/` directory contains the aggregated outputs that support the paper's claims. These are produced by the scripts above and serve as the single source of truth for all numbers, tables, and figures in the paper.

## Dependencies

```
pip install matplotlib numpy scipy
```

## Benchmark

All analysis is based on SWE-bench Verified (https://github.com/princeton-nlp/SWE-bench), a curated benchmark of 500 real-world GitHub issues. Nine LLMs spanning three capability tiers (large, mid, small) are each executed independently 6 times per bug using SWE-agent.
