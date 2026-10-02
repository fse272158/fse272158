#!/usr/bin/env python3
"""
Within-model convergence analysis.

For each model independently:
- For each instance_id, collect patches from all runs where it appears
- Compare patches across runs using a given similarity method
- Compute convergence score: (runs with "same" patch) / (total runs present)
- Save per-bug convergence labels as JSONL

Each record tracks two correctness metrics:
- resolved_any: resolved in at least one run (lenient)
- resolved_all: resolved in every run that has data (strict, primary metric)
"""

import analysis_config as cfg
import analysis_data_io as io
import analysis_similarity as sim


def compute_convergence_for_model(
    model: str,
    method: str,
    force: bool = False,
) -> list[dict]:
    """
    Compute within-model convergence for a single model across all runs.

    For each instance_id:
      1. Load patches from all runs where it appears
      2. Use the first run's patch as reference
      3. Compare reference to every other run's patch using the method
      4. convergence = n_agree / n_runs_present

    Returns a list of per-bug result dicts.
    """
    output_path = cfg.CONVERGENCE_JSONL[method]

    # Check if we already have results for this model+method
    if not force and io.intermediate_exists(output_path):
        existing = io.load_jsonl(output_path)
        model_records = [r for r in existing if r["model"] == model]
        if model_records:
            print(f"[skip] {model} / {method}: {len(model_records)} records already exist")
            return model_records

    # Collect all instance_ids across all runs for this model
    all_instance_ids = set()
    for run_id in cfg.RUNS:
        ids = io.load_all_instance_ids(run_id, model)
        all_instance_ids.update(ids)

    # Load patches and reports for all runs
    patches = {}
    reports = {}
    for run_id in cfg.RUNS:
        for inst_id in all_instance_ids:
            patches[(run_id, inst_id)] = io.load_patch(run_id, model, inst_id)
            reports[(run_id, inst_id)] = io.load_report(run_id, model, inst_id)

    # Compute convergence per bug
    results = []
    for inst_id in sorted(all_instance_ids):
        # Find runs where this instance has a patch
        runs_with_data = []
        for run_id in cfg.RUNS:
            if patches.get((run_id, inst_id)) is not None:
                runs_with_data.append(run_id)

        if len(runs_with_data) < 2:
            continue

        # Use the first run's patch as reference
        ref_run = runs_with_data[0]
        ref_patch = patches[(ref_run, inst_id)]
        ref_report = reports[(ref_run, inst_id)]

        # Compare reference to every other run
        n_agree = 0
        for run_id in runs_with_data:
            cmp_patch = patches[(run_id, inst_id)]
            cmp_report = reports[(run_id, inst_id)]
            if sim.compare_patches(ref_patch, cmp_patch, ref_report, cmp_report, method):
                n_agree += 1

        n_runs = len(runs_with_data)
        convergence = round(n_agree / n_runs, 4)
        bucket = _get_bucket(n_agree, n_runs)

        # Determine resolved status: both ANY and ALL
        resolved_any = False
        resolved_all = True
        n_resolved = 0
        for run_id in runs_with_data:
            report = reports.get((run_id, inst_id))
            is_resolved = report is not None and _is_resolved(report)
            if is_resolved:
                resolved_any = True
                n_resolved += 1
            else:
                resolved_all = False

        # If no runs have a report at all, resolved_all should be False
        if not resolved_any:
            resolved_all = False

        results.append({
            "instance_id": inst_id,
            "model": model,
            "method": method,
            "n_runs": n_runs,
            "n_agree": n_agree,
            "convergence": convergence,
            "bucket": bucket,
            "resolved_any": resolved_any,
            "resolved_all": resolved_all,
            "n_resolved": n_resolved,
        })

    return results


def save_convergence_results(results: list[dict], method: str) -> None:
    """
    Save convergence results for a given method.
    Merges with existing results for other models (if any).
    """
    output_path = cfg.CONVERGENCE_JSONL[method]

    model = results[0]["model"] if results else None

    existing = io.load_jsonl(output_path)
    if existing and model:
        existing = [r for r in existing if r.get("model") != model]

    all_results = existing + results
    io.save_jsonl(all_results, output_path)
    print(f"[ok] Saved {len(results)} records for {model} / {method} -> {output_path}")


def _get_bucket(n_agree: int, n_runs: int) -> str:
    """Map (n_agree, n_runs) to a convergence bucket label.

    For 6 runs: use standard buckets (6/6, 5/6, 4/6, 3/6, 2/6, 1/6)
    For other run counts: map to equivalent 6-run bucket based on convergence rate
    """
    if n_runs == 6:
        if n_agree == 6:
            return "6/6"
        elif n_agree == 5:
            return "5/6"
        elif n_agree == 4:
            return "4/6"
        elif n_agree == 3:
            return "3/6"
        elif n_agree == 2:
            return "2/6"
        else:
            return "1/6"
    else:
        rate = n_agree / n_runs if n_runs > 0 else 0
        if rate >= 5/6:
            return "5/6"
        elif rate >= 4/6:
            return "4/6"
        elif rate >= 3/6:
            return "3/6"
        elif rate >= 2/6:
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
