#!/usr/bin/env python3
"""
Main orchestrator for convergence-correctness analysis.

Incremental processing pipeline:
1. For each similarity method (exact, syntactic, semantic):
   a. Compute within-model convergence for gpt-5-mini and claude-opus-4-7
   b. Compute cross-model agreement for all 3 model pairs
2. Compute aggregate statistics (precision per bucket)
3. Generate paper-ready plots and CSV tables

Each step saves intermediate JSONL/JSON files and skips if already present.
Use --force to recompute.
"""

import sys
import argparse

import analysis_config as cfg
import analysis_convergence as conv
import analysis_cross_model as cm
import analysis_stats as stats
import analysis_plot as plots


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_convergence(method: str, force: bool = False) -> None:
    """Compute within-model convergence for both models."""
    print(f"\n--- Convergence: {method} ---")
    for model in cfg.WITHIN_MODEL_MODELS:
        print(f"  {model}...")
        results = conv.compute_convergence_for_model(model, method, force=force)
        if results:
            conv.save_convergence_results(results, method)
        else:
            print(f"    [skip] {model}: no data")


def step_cross_model(method: str, force: bool = False) -> None:
    """Compute cross-model agreement for all pairs."""
    print(f"\n--- Cross-Model: {method} ---")
    for model_a, model_b in cfg.CROSS_MODEL_PAIRS:
        print(f"  {model_a} vs {model_b}...")
        results = cm.compute_cross_model_for_pair(model_a, model_b, method, force=force)
        if results:
            cm.save_cross_model_results(results, method)
        else:
            print(f"    [skip] {model_a} vs {model_b}: no common data")


def step_stats(force: bool = False) -> None:
    """Compute aggregate statistics."""
    print("\n--- Statistics ---")
    stats.run_stats(force=force)


def step_plots() -> None:
    """Generate plots."""
    print("\n--- Plots ---")
    plots.run_plots()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convergence-correctness analysis")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all steps (ignore existing intermediates)")
    parser.add_argument("--only", choices=["convergence", "cross-model", "stats", "plots"],
                        help="Run only a specific step")
    parser.add_argument("--method", choices=cfg.SIMILARITY_METHODS + ["all"],
                        default="all", help="Which similarity method(s) to process")
    args = parser.parse_args()

    force = args.force
    only = args.only
    methods = cfg.SIMILARITY_METHODS if args.method == "all" else [args.method]

    # Ensure analysis directory exists
    cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONVERGENCE-CORRECTNESS ANALYSIS")
    print(f"Methods: {methods}")
    print(f"Force: {force}")
    print(f"Only: {only or 'all'}")
    print("=" * 60)

    # Step 1: Within-model convergence
    if only in (None, "convergence"):
        for method in methods:
            step_convergence(method, force)

    # Step 2: Cross-model agreement
    if only in (None, "cross-model"):
        for method in methods:
            step_cross_model(method, force)

    # Step 3: Statistics
    if only in (None, "stats"):
        step_stats(force)

    # Step 4: Plots
    if only in (None, "plots"):
        step_plots()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Outputs in: {cfg.ANALYSIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()