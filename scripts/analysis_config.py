#!/usr/bin/env python3
"""
Single source of truth for all analysis configuration.
All hard-coded values go here. No other file should contain hard-coded paths,
model names, run numbers, or thresholds.

Output files:
- Intermediates (for regeneration, not in paper): convergence_*.jsonl, cross_model_*.jsonl, stats_*.json, cross_model_stats.json
- Paper files (prefixed with paper_): paper_convergence_bars.png, paper_cross_model_bars.csv, paper_convergence_precision.csv, paper_cross_model_agreement.csv
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all derived from PROJECT_ROOT so the repo is location-independent
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = PROJECT_ROOT / "dataset" / "run_test"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# ---------------------------------------------------------------------------
# Models and runs
# ---------------------------------------------------------------------------
WITHIN_MODEL_MODELS = ["gpt-5", "claude-opus-4-7", "claude-4-sonnet", "gpt-4o", "claude-3-5-sonnet", "devstral-2-123b", "kimi-k2-instruct", "qwen3-32b", "gpt-5-mini"]
RUNS = [1, 2, 3, 4, 5, 6]

MODEL_TIERS = {
    "large": ["gpt-5", "claude-opus-4-7"],
    "mid": ["claude-4-sonnet", "gpt-4o", "claude-3-5-sonnet", "devstral-2-123b"],
    "small": ["kimi-k2-instruct", "qwen3-32b", "gpt-5-mini"],
}
MODEL_TO_TIER = {m: t for t, models in MODEL_TIERS.items() for m in models}

# Cross-model agreement: within-tier pairs
CROSS_MODEL_PAIRS = [
    ("gpt-5", "claude-opus-4-7"),
    ("gpt-4o", "claude-3-5-sonnet"),
    ("gpt-4o", "claude-4-sonnet"),
    ("claude-3-5-sonnet", "claude-4-sonnet"),
    ("devstral-2-123b", "gpt-4o"),
    ("devstral-2-123b", "claude-3-5-sonnet"),
    ("devstral-2-123b", "claude-4-sonnet"),
    ("kimi-k2-instruct", "qwen3-32b"),
    ("kimi-k2-instruct", "gpt-5-mini"),
    ("qwen3-32b", "gpt-5-mini"),
]

# ---------------------------------------------------------------------------
# Similarity methods
# ---------------------------------------------------------------------------
SIMILARITY_METHODS = ["exact", "syntactic", "semantic"]
# File name keys match the public method names
_METHOD_FILE_KEY = {"exact": "exact", "syntactic": "syntactic", "semantic": "semantic"}

# ---------------------------------------------------------------------------
# Convergence buckets
# ---------------------------------------------------------------------------
CONVERGENCE_BUCKETS = ["6/6", "5/6", "4/6", "3/6", "2/6", "1/6"]

# ---------------------------------------------------------------------------
# Output file names (keyed by public method name, file names use internal keys)
# ---------------------------------------------------------------------------
CONVERGENCE_JSONL = {
    method: ANALYSIS_DIR / f"convergence_{_METHOD_FILE_KEY[method]}.jsonl"
    for method in SIMILARITY_METHODS
}

CROSS_MODEL_JSONL = {
    method: ANALYSIS_DIR / f"cross_model_{_METHOD_FILE_KEY[method]}.jsonl"
    for method in SIMILARITY_METHODS
}

STATS_JSON = {
    method: ANALYSIS_DIR / f"stats_{_METHOD_FILE_KEY[method]}.json"
    for method in SIMILARITY_METHODS
}

CROSS_MODEL_STATS_JSON = ANALYSIS_DIR / "cross_model_stats.json"

# Paper files (renamed with paper_ prefix)
PAPER_PLOT_CONVERGENCE_BARS = ANALYSIS_DIR / "paper_convergence_bars.png"
PAPER_PLOT_CROSS_MODEL_BARS = ANALYSIS_DIR / "paper_cross_model_bars.png"
PAPER_CONVERGENCE_PRECISION_CSV = ANALYSIS_DIR / "paper_convergence_precision.csv"
PAPER_CROSS_MODEL_AGREEMENT_CSV = ANALYSIS_DIR / "paper_cross_model_agreement.csv"

SENSITIVITY_N_RESULTS = ANALYSIS_DIR / "sensitivity_n_results.json"
REFERENCE_ABLATION_RESULTS = ANALYSIS_DIR / "reference_ablation_results.json"
EARLY_STOPPING_BASELINE_RESULTS = ANALYSIS_DIR / "early_stopping_baseline_results.json"
PRECISION_RECALL_COMPARISON_JSON = ANALYSIS_DIR / "precision_recall_comparison.json"
PRECISION_RECALL_COMPARISON_CSV = ANALYSIS_DIR / "precision_recall_comparison.csv"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TOTAL_INSTANCES = 500

# ---------------------------------------------------------------------------
# Plot font sizes (paper figures only: complementarity, cost-effectiveness)
# Change these to resize all text in the 3 paper figures at once.
# ---------------------------------------------------------------------------
PLOT_FONT_TITLE = 40
PLOT_FONT_AXIS_LABEL = 35
PLOT_FONT_TICK = 20
PLOT_FONT_LEGEND = 25
PLOT_FONT_BAR_INSIDE = 20
PLOT_FONT_BAR_ABOVE = 20
PLOT_FONT_COUNT_ANNOTATION = 30
