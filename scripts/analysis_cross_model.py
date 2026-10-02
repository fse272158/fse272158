#!/usr/bin/env python3
"""
Cross-model agreement analysis — set-based comparison.

For each pair of models (e.g., gpt-5-mini vs claude-opus-4-7):
- Collect ALL patches from model A across all its runs
- Collect ALL patches from model B across all its runs
- For each run in model A, check if ANY patch from model B agrees
- Bucket = (number of model A runs that have at least one match in model B) / 6

This maps directly to within-model buckets:
- 1/6 = only 1 run from model A found a matching patch in model B
- 6/6 = all 6 runs from model A found a matching patch in model B

For the "resolved" metric: if models agree on a patch, that patch either
resolves the bug or it doesn't. So "resolved" = the agreeing patch resolved.
"""

import analysis_config as cfg
import analysis_data_io as io
import analysis_similarity as sim


def get_all_runs_for_model(model: str) -> list[int]:
    """Find all runs where a model has evaluation data."""
    runs = []
    for run_id in cfg.RUNS:
        summary = io.load_summary(run_id, model)
        if summary is not None:
            runs.append(run_id)
    return runs


def compute_cross_model_for_pair(
    model_a: str,
    model_b: str,
    method: str,
    force: bool = False,
) -> list[dict]:
    """
    Compute cross-model agreement for a pair of models using set-based comparison.

    Bucket definition (matching within-model 1/6-6/6 scale):
    - For each run in model A, check if ANY patch from any run in model B matches
    - n_matching = number of model A runs with at least one match in model B
    - bucket = n_matching / 6

    This means:
    - 1/6 = only 1 of model A's runs found a match in model B's runs
    - 6/6 = all 6 of model A's runs found a match in model B's runs

    "Resolved" = at least one agreeing pair resolved the bug.
    """
    pair_key = f"{model_a}_vs_{model_b}"
    output_path = cfg.CROSS_MODEL_JSONL[method]

    # Check existing
    if not force and io.intermediate_exists(output_path):
        existing = io.load_jsonl(output_path)
        pair_records = [r for r in existing if r.get("pair") == pair_key]
        if pair_records:
            print(f"[skip] {pair_key} / {method}: {len(pair_records)} records already exist")
            return pair_records

    # Get all runs for each model
    runs_a = get_all_runs_for_model(model_a)
    runs_b = get_all_runs_for_model(model_b)

    if not runs_a or not runs_b:
        print(f"[warn] {pair_key}: no data found")
        return []

    # Collect all instance_ids across all runs for both models
    all_instance_ids = set()
    for run_id in runs_a:
        all_instance_ids.update(io.load_all_instance_ids(run_id, model_a))
    for run_id in runs_b:
        all_instance_ids.update(io.load_all_instance_ids(run_id, model_b))

    # Load all patches and reports
    patches_a = {}
    patches_b = {}
    reports_a = {}
    reports_b = {}

    for run_id in runs_a:
        patches_a[run_id] = {}
        reports_a[run_id] = {}
        for inst_id in all_instance_ids:
            patches_a[run_id][inst_id] = io.load_patch(run_id, model_a, inst_id)
            reports_a[run_id][inst_id] = io.load_report(run_id, model_a, inst_id)

    for run_id in runs_b:
        patches_b[run_id] = {}
        reports_b[run_id] = {}
        for inst_id in all_instance_ids:
            patches_b[run_id][inst_id] = io.load_patch(run_id, model_b, inst_id)
            reports_b[run_id][inst_id] = io.load_report(run_id, model_b, inst_id)

    # Compute agreement per bug
    results = []
    for inst_id in sorted(all_instance_ids):
        # Collect all patches for this instance from each model
        patches_a_list = []  # [(run_id, patch_text, report), ...]
        patches_b_list = []

        for run_id in runs_a:
            patch = patches_a[run_id].get(inst_id)
            report = reports_a[run_id].get(inst_id)
            if patch is not None:
                patches_a_list.append((run_id, patch, report))

        for run_id in runs_b:
            patch = patches_b[run_id].get(inst_id)
            report = reports_b[run_id].get(inst_id)
            if patch is not None:
                patches_b_list.append((run_id, patch, report))

        if not patches_a_list or not patches_b_list:
            continue

        # For each run in A, check if ANY patch from B matches
        n_matching = 0
        resolved_in_matching = False

        for run_a, patch_a, report_a in patches_a_list:
            found_match = False
            for run_b, patch_b, report_b in patches_b_list:
                if sim.compare_patches(patch_a, patch_b, report_a, report_b, method):
                    found_match = True
                    if report_a is not None and _is_resolved(report_a):
                        resolved_in_matching = True
            if found_match:
                n_matching += 1

        # Bucket based on how many of model A's runs found a match
        bucket = _get_bucket_label(n_matching)

        results.append({
            "instance_id": inst_id,
            "pair": pair_key,
            "model_a_name": model_a,
            "model_b_name": model_b,
            "method": method,
            "n_runs_a": len(patches_a_list),
            "n_runs_b": len(patches_b_list),
            "n_matching": n_matching,
            "bucket": bucket,
            "resolved": resolved_in_matching,
        })

    return results


def save_cross_model_results(results: list[dict], method: str) -> None:
    """Save cross-model results for a given method."""
    output_path = cfg.CROSS_MODEL_JSONL[method]
    pair_key = results[0]["pair"] if results else None

    existing = io.load_jsonl(output_path)
    if existing and pair_key:
        existing = [r for r in existing if r.get("pair") != pair_key]

    all_results = existing + results
    io.save_jsonl(all_results, output_path)
    print(f"[ok] Saved {len(results)} records for {pair_key} / {method} -> {output_path}")


def _get_bucket_label(n_matching: int) -> str:
    """Map n_matching (0-6) to bucket label matching within-model scale."""
    if n_matching >= 6:
        return "6/6"
    elif n_matching == 5:
        return "5/6"
    elif n_matching == 4:
        return "4/6"
    elif n_matching == 3:
        return "3/6"
    elif n_matching == 2:
        return "2/6"
    else:
        return "1/6"


def _is_resolved(report: dict) -> bool:
    """Check if a report indicates the instance was resolved."""
    if not isinstance(report, dict):
        return False
    instance_data = next(iter(report.values()), None)
    if not isinstance(instance_data, dict):
        return False
    return bool(instance_data.get("resolved", False))
