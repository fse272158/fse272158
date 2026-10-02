#!/usr/bin/env python3
"""
Aggregate statistics: precision per convergence bucket, cross-model correctness,
and confidence intervals.

Reads intermediate JSONL files and computes:
- Per-method, per-model convergence stats: precision (resolved/total) per bucket
- Cross-model agreement stats: correctness by agreement bucket
- Wilson score confidence intervals for all proportions
- Saves as JSON and generates paper-ready CSV tables (paper_*.csv)
"""

import math
import csv
from collections import defaultdict

import analysis_config as cfg
import analysis_data_io as io


# ---------------------------------------------------------------------------
# Confidence intervals (Wilson score interval for binomial proportion)
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lower, upper) bounds.
    z=1.96 for 95% CI.
    """
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Convergence stats (within-model) — PER MODEL
# ---------------------------------------------------------------------------

def compute_convergence_stats(method: str) -> dict:
    """
    Compute precision per convergence bucket for each model separately.

    Reads convergence_{method}.jsonl, groups by (model, bucket), computes:
      - total bugs in bucket
      - resolved_any: resolved in at least one run
      - resolved_all: resolved in every run
      - precision_any = resolved_any / total
      - precision_all = resolved_all / total
      - Wilson 95% CI for both
    """
    path = cfg.CONVERGENCE_JSONL[method]
    records = io.load_jsonl(path)

    # bucket_stats[model][bucket] = {total, resolved_any, resolved_all}
    bucket_stats: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"total": 0, "resolved_any": 0, "resolved_all": 0}
    ))

    for r in records:
        model = r["model"]
        bucket = r["bucket"]
        bucket_stats[model][bucket]["total"] += 1
        if r.get("resolved_any", r.get("resolved", False)):
            bucket_stats[model][bucket]["resolved_any"] += 1
        if r.get("resolved_all", False):
            bucket_stats[model][bucket]["resolved_all"] += 1

    # Build results per model
    results = {}
    for model in cfg.WITHIN_MODEL_MODELS:
        model_results = {}
        total_bugs = 0
        total_resolved_any = 0
        total_resolved_all = 0

        for bucket in cfg.CONVERGENCE_BUCKETS:
            stats = bucket_stats[model].get(bucket)
            if not stats or stats["total"] == 0:
                continue
            total = stats["total"]
            res_any = stats["resolved_any"]
            res_all = stats["resolved_all"]
            prec_any = round(res_any / total, 4)
            prec_all = round(res_all / total, 4)
            ci_any = wilson_ci(res_any, total)
            ci_all = wilson_ci(res_all, total)

            model_results[bucket] = {
                "total": total,
                "resolved_any": res_any,
                "resolved_all": res_all,
                "precision_any": prec_any,
                "precision_all": prec_all,
                "ci_any_lower": round(ci_any[0], 4),
                "ci_any_upper": round(ci_any[1], 4),
                "ci_all_lower": round(ci_all[0], 4),
                "ci_all_upper": round(ci_all[1], 4),
            }
            total_bugs += total
            total_resolved_any += res_any
            total_resolved_all += res_all

        # Overall stats
        overall_prec_any = round(total_resolved_any / total_bugs, 4) if total_bugs > 0 else 0.0
        overall_prec_all = round(total_resolved_all / total_bugs, 4) if total_bugs > 0 else 0.0
        model_results["overall"] = {
            "total_bugs": total_bugs,
            "total_resolved_any": total_resolved_any,
            "total_resolved_all": total_resolved_all,
            "overall_precision_any": overall_prec_any,
            "overall_precision_all": overall_prec_all,
            "ci_any_lower": round(wilson_ci(total_resolved_any, total_bugs)[0], 4),
            "ci_any_upper": round(wilson_ci(total_resolved_any, total_bugs)[1], 4),
            "ci_all_lower": round(wilson_ci(total_resolved_all, total_bugs)[0], 4),
            "ci_all_upper": round(wilson_ci(total_resolved_all, total_bugs)[1], 4),
        }
        results[model] = model_results

    results["method"] = method
    return results


def save_convergence_stats(stats: dict, method: str) -> None:
    """Save convergence stats to JSON."""
    output_path = cfg.STATS_JSON[method]
    io.save_json(stats, output_path)
    print(f"[ok] Saved convergence stats for {method} -> {output_path}")


# ---------------------------------------------------------------------------
# Cross-model stats
# ---------------------------------------------------------------------------

def compute_cross_model_stats(method: str) -> dict:
    """
    Compute cross-model agreement stats for a given method.

    Reads cross_model_{method}.jsonl, groups by (pair, bucket), computes:
      - total bugs in bucket
      - resolved: bugs where the agreeing patch resolved the bug

    For set-based cross-model comparison with 1/6-6/6 buckets:
    - bucket = n_matching / 6 (how many runs from model A found a match in model B)
    - resolved = at least one agreeing pair resolved the bug
    """
    path = cfg.CROSS_MODEL_JSONL[method]
    records = io.load_jsonl(path)

    # Group by (pair, bucket)
    bucket_stats = {}
    for r in records:
        pair = r.get("pair", "unknown")
        bucket = r.get("bucket", "unknown")
        key = (pair, bucket)

        if key not in bucket_stats:
            bucket_stats[key] = {
                "total": 0,
                "resolved": 0,
                "model_a_name": r.get("model_a_name", "?"),
                "model_b_name": r.get("model_b_name", "?"),
            }

        s = bucket_stats[key]
        s["total"] += 1
        if r.get("resolved", False):
            s["resolved"] += 1

    # Build results
    results = {}
    for (pair, bucket), s in bucket_stats.items():
        total = s["total"]
        res = s["resolved"]

        results[(pair, bucket)] = {
            "pair": pair,
            "bucket": bucket,
            "total": total,
            "resolved": res,
            "unresolved": total - res,
            "resolved_rate": round(res / total, 4) if total > 0 else 0.0,
            "model_a_name": s["model_a_name"],
            "model_b_name": s["model_b_name"],
        }

    results["method"] = method
    return results


def save_cross_model_stats(stats: dict, method: str) -> None:
    """Save cross-model stats, merging with other methods."""
    output_path = cfg.CROSS_MODEL_STATS_JSON
    existing = io.load_json(output_path) or {}

    # Convert tuple keys to strings for JSON serialization
    serializable_stats = {}
    for key, value in stats.items():
        if isinstance(key, tuple):
            serializable_stats[f"{key[0]}|{key[1]}"] = value
        else:
            serializable_stats[key] = value

    existing[method] = serializable_stats
    io.save_json(existing, output_path)
    print(f"[ok] Saved cross-model stats for {method} -> {output_path}")




# ---------------------------------------------------------------------------
# Paper-ready CSV tables
# ---------------------------------------------------------------------------

BUCKET_ORDER = ["6/6", "5/6", "4/6", "3/6", "2/6", "1/6"]


def generate_paper_tables() -> None:
    """Generate paper-ready CSV tables."""
    convergence = {}
    cross_model = {}

    for method in cfg.SIMILARITY_METHODS:
        stats_path = cfg.STATS_JSON[method]
        if stats_path.is_file():
            convergence[method] = io.load_json(stats_path)

    cross_path = cfg.CROSS_MODEL_STATS_JSON
    if cross_path.is_file():
        cross_model = io.load_json(cross_path)

    # Table 1: Convergence precision by bucket (per model, per method)
    _write_convergence_precision_csv(convergence)

    # Table 2: Cross-model agreement (per method)
    _write_cross_model_agreement_csv(cross_model)


def _write_convergence_precision_csv(convergence: dict) -> None:
    """Table 1: Convergence precision per bucket, per model, per method."""
    rows = []
    for method in cfg.SIMILARITY_METHODS:
        method_data = convergence.get(method, {})
        for model in cfg.WITHIN_MODEL_MODELS:
            model_data = method_data.get(model, {})
            for bucket in BUCKET_ORDER:
                if bucket in model_data and bucket != "overall":
                    d = model_data[bucket]
                    rows.append({
                        "method": method,
                        "model": model,
                        "bucket": bucket,
                        "total": d["total"],
                        "resolved_any": d["resolved_any"],
                        "resolved_all": d["resolved_all"],
                        "precision_any": d["precision_any"],
                        "precision_all": d["precision_all"],
                        "ci_any_lower": d["ci_any_lower"],
                        "ci_any_upper": d["ci_any_upper"],
                        "ci_all_lower": d["ci_all_lower"],
                        "ci_all_upper": d["ci_all_upper"],
                    })

    path = cfg.PAPER_CONVERGENCE_PRECISION_CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[ok] Saved convergence precision table -> {path} ({len(rows)} rows)")


def _write_cross_model_agreement_csv(cross_model: dict) -> None:
    """Table 2: Cross-model agreement per pair, per bucket, per method.

    For set-based cross-model comparison:
    - bucket: agreement level (Full/High/Low/None)
    - has_agreement: bugs with at least one agreeing pair
    - resolved_in_agreeing: bugs where the agreeing patch resolved the bug
    """
    # Bucket order for display
    bucket_order = ["Full", "High", "Low", "None"]

    # Bucket order for display (matching within-model: 1/6 to 6/6)
    bucket_order = ["1/6", "2/6", "3/6", "4/6", "5/6", "6/6"]

    rows = []
    for method in cfg.SIMILARITY_METHODS:
        method_data = cross_model.get(method, {})
        for key, d in method_data.items():
            if key == "method":
                continue
            # Keys may be tuples or stringified "pair|bucket" from JSON
            if isinstance(key, tuple):
                pair, bucket = key
            else:
                pair, bucket = key.split("|", 1)
            rows.append({
                "method": method,
                "pair": pair,
                "bucket": bucket,
                "total": d["total"],
                "resolved": d["resolved"],
                "unresolved": d["unresolved"],
                "resolved_rate": d["resolved_rate"],
                "model_a_name": d["model_a_name"],
                "model_b_name": d["model_b_name"],
            })

    # Sort by method, pair, bucket order
    def sort_key(r):
        bucket_idx = bucket_order.index(r["bucket"]) if r["bucket"] in bucket_order else 999
        return (r["method"], r["pair"], bucket_idx)

    rows.sort(key=sort_key)

    path = cfg.PAPER_CROSS_MODEL_AGREEMENT_CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[ok] Saved cross-model agreement table -> {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main orchestrator for stats
# ---------------------------------------------------------------------------

def run_stats(force: bool = False) -> None:
    """Run all stats computation."""
    print("=== Computing convergence stats (per model) ===")
    for method in cfg.SIMILARITY_METHODS:
        stats = compute_convergence_stats(method)
        save_convergence_stats(stats, method)

    print("=== Computing cross-model stats ===")
    for method in cfg.SIMILARITY_METHODS:
        stats = compute_cross_model_stats(method)
        save_cross_model_stats(stats, method)

    print("=== Generating paper tables ===")
    generate_paper_tables()


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    run_stats(force)
