#!/usr/bin/env python3
"""
M9: Apply multiple testing correction to Fisher's exact test results.

Takes the existing stats_test_results.json and adds Bonferroni and
Benjamini-Hochberg corrected p-values.

Outputs:
  analysis/stats_test_results_corrected.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analysis_config as cfg
import analysis_data_io as io


def bonferroni_correction(p_values: list[float], n_tests: int) -> list[float]:
    """Bonferroni correction: multiply each p-value by n_tests, cap at 1.0."""
    return [min(p * n_tests, 1.0) for p in p_values]


def benjamini_hochberg_correction(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    prev = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed), 1):
        bh_value = min(p * n / rank, prev)
        corrected[orig_idx] = bh_value
        prev = bh_value
    return corrected


def main():
    print("=== Multiple Testing Correction (M9) ===\n")

    # Load existing test results
    results = io.load_json(cfg.ANALYSIS_DIR / 'stats_test_results.json')
    if not results:
        print("ERROR: stats_test_results.json not found. Run analysis_stats_tests.py first.")
        sys.exit(1)

    # Extract p-values in order
    test_names = list(results.keys())
    p_values = [results[name]['p_value'] for name in test_names]
    n_tests = len(p_values)

    print(f"Number of tests: {n_tests}")
    print(f"Raw p-values: {[f'{p:.2e}' for p in p_values]}")

    # Apply corrections
    bonf = bonferroni_correction(p_values, n_tests)
    bh = benjamini_hochberg_correction(p_values)

    # Add to results
    for i, name in enumerate(test_names):
        results[name]['p_bonferroni'] = bonf[i]
        results[name]['p_bh'] = bh[i]
        results[name]['significant_bonferroni_05'] = bonf[i] < 0.05
        results[name]['significant_bh_05'] = bh[i] < 0.05

    # Save corrected results
    out_path = cfg.ANALYSIS_DIR / 'stats_test_results_corrected.json'
    io.save_json(results, out_path)
    print(f"\nSaved corrected results: {out_path}")

    # Print summary table
    print(f"\n{'Test':<45s} {'Raw p':>10s} {'Bonf':>10s} {'BH':>10s} {'B<0.05':>7s} {'BH<0.05':>8s}")
    print("-" * 95)
    for name in test_names:
        r = results[name]
        raw = f"{r['p_value']:.2e}"
        b = f"{r['p_bonferroni']:.2e}"
        bh_v = f"{r['p_bh']:.2e}"
        sig_b = '✓' if r['significant_bonferroni_05'] else '✗'
        sig_bh = '✓' if r['significant_bh_05'] else '✗'
        print(f"  {name:<43s} {raw:>10s} {b:>10s} {bh_v:>10s} {sig_b:>7s} {sig_bh:>8s}")

    all_bonf = all(r['significant_bonferroni_05'] for r in results.values())
    all_bh = all(r['significant_bh_05'] for r in results.values())
    print(f"\nAll significant after Bonferroni: {all_bonf}")
    print(f"All significant after BH: {all_bh}")
    print("Done.")


if __name__ == '__main__':
    main()
