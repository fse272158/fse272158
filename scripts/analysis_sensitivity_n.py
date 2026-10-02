#!/usr/bin/env python3
"""
M15: Sensitivity analysis for the choice of N (number of runs).

The main paper uses N=6 for convergence analysis. This script evaluates
whether the convergence-correctness correlation is robust to the choice of N
by varying N from 3 to 6 and measuring:
  1. Precision at the highest convergence bucket (N/N) and lowest (1/N)
  2. Contrast (precision delta between N/N and 1/N)
  3. Sample sizes at the extremes

For each N, use the first N runs from the existing 6 runs per model.

Outputs:
  analysis/sensitivity_n_results.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analysis_config as cfg
import analysis_data_io as io
import analysis_similarity as sim
import analysis_stats as stats


def compute_convergence_variable_n(instance_id: str, model: str,
                                    available_runs: list[int], n: int,
                                    method: str) -> dict | None:
    """
    Compute convergence for a bug using exactly N runs (first N from available).
    Returns dict with bucket, resolved_all, etc., or None if insufficient data.
    """
    runs_to_use = available_runs[:n]
    if len(runs_to_use) < n:
        return None

    # Check all N runs have data
    patches = []
    reports = []
    for run_id in runs_to_use:
        p = io.load_patch(run_id, model, instance_id)
        r = io.load_report(run_id, model, instance_id)
        if p is None or r is None:
            return None
        patches.append(p)
        reports.append(r)

    # Reference = first run
    agree = 1  # reference agrees with itself
    for i in range(1, n):
        if method == 'syntactic':
            if sim.syntactic_match(patches[0], patches[i]):
                agree += 1
        elif method == 'semantic':
            if sim.semantic_match(reports[0], reports[i]):
                agree += 1
        elif method == 'exact':
            if sim.exact_match(patches[0], patches[i]):
                agree += 1

    bucket = f"{agree}/{n}"
    all_resolved = all(
        r.get(instance_id, {}).get('resolved', False) or
        next(iter(r.values()), {}).get('resolved', False)
        for r in reports
    )

    return {
        'instance_id': instance_id,
        'model': model,
        'n': n,
        'agree': agree,
        'bucket': bucket,
        'resolved_all': all_resolved,
    }


def main():
    print("=== Sensitivity Analysis: Varying N (number of runs) ===\n")

    models_runs = {model: list(cfg.RUNS) for model in cfg.WITHIN_MODEL_MODELS}
    method = 'syntactic'

    results = {}
    for model, available_runs in models_runs.items():
        max_n = len(available_runs)
        print(f"\n--- Model: {model} (up to {max_n} runs available) ---")

        # Get instance IDs that have data for all max_n runs
        instance_sets = []
        for run_id in available_runs:
            ids = set(io.load_all_instance_ids(run_id, model))
            instance_sets.append(ids)
        complete_ids = sorted(set.intersection(*instance_sets))
        print(f"  Bugs with all {max_n} runs: {len(complete_ids)}")

        model_results = {}
        for n in range(3, max_n + 1):
            bucket_stats = {}  # bucket -> {total, resolved}
            for iid in complete_ids:
                r = compute_convergence_variable_n(iid, model, available_runs,
                                                    n, method)
                if r is None:
                    continue
                bucket = r['bucket']
                if bucket not in bucket_stats:
                    bucket_stats[bucket] = {'total': 0, 'resolved': 0}
                bucket_stats[bucket]['total'] += 1
                if r['resolved_all']:
                    bucket_stats[bucket]['resolved'] += 1

            # Compute precision at N/N (full convergence) and 1/N (min convergence)
            full_bucket = f"{n}/{n}"
            min_bucket = f"1/{n}"

            n_n_total = bucket_stats.get(full_bucket, {}).get('total', 0)
            n_n_resolved = bucket_stats.get(full_bucket, {}).get('resolved', 0)
            n_n_prec = n_n_resolved / n_n_total if n_n_total > 0 else None

            one_n_total = bucket_stats.get(min_bucket, {}).get('total', 0)
            one_n_resolved = bucket_stats.get(min_bucket, {}).get('resolved', 0)
            one_n_prec = one_n_resolved / one_n_total if one_n_total > 0 else None

            contrast = None
            if n_n_prec is not None and one_n_prec is not None:
                contrast = round(n_n_prec - one_n_prec, 4)

            model_results[f'N={n}'] = {
                'full_bucket': full_bucket,
                'full_precision': round(n_n_prec, 4) if n_n_prec is not None else None,
                'full_n': n_n_total,
                'min_bucket': min_bucket,
                'min_precision': round(one_n_prec, 4) if one_n_prec is not None else None,
                'min_n': one_n_total,
                'contrast': contrast,
                'bucket_details': bucket_stats,
            }

            # Print summary
            n_n_ci = stats.wilson_ci(n_n_resolved, n_n_total) if n_n_total > 0 else (None, None)
            one_n_ci = stats.wilson_ci(one_n_resolved, one_n_total) if one_n_total > 0 else (None, None)
            n_n_str = f"{n_n_prec:.1%}" if n_n_prec is not None else "N/A"
            one_n_str = f"{one_n_prec:.1%}" if one_n_prec is not None else "N/A"
            contrast_str = f"{contrast:.1%}" if contrast is not None else "N/A"
            print(f"  N={n}: {full_bucket} prec={n_n_str} (n={n_n_total}), "
                  f"{min_bucket} prec={one_n_str} (n={one_n_total}), "
                  f"contrast={contrast_str}")

        results[model] = model_results

    # Save results
    output = {
        'method': method,
        'results': results,
    }
    out_path = cfg.ANALYSIS_DIR / 'sensitivity_n_results.json'
    io.save_json(output, out_path)
    print(f"\nSaved: {out_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
