#!/usr/bin/env python3
"""
Convergence to Incorrect Patches — Characterization Analysis

For bugs where a model produces the same patch across all runs (6/6 convergence)
but the bug is NOT resolved, we characterize:
1. Which bugs show convergence to incorrect patches?
2. What are their characteristics? (repo, patch size, test failure patterns)
3. What do the incorrect converged patches look like? (FAIL_TO_PASS gaps, regressions)
4. How do they compare to correctly resolved converged bugs?

Output:
- analysis/convergence_to_incorrect_patches.jsonl       (per-bug details)
- analysis/convergence_to_incorrect_patches_summary.txt (text summary for paper)
- analysis/convergence_to_incorrect_patches_repos.csv   (repo-level breakdown)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import analysis_config as cfg
import analysis_data_io as io


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_test_summary(report: dict) -> dict:
    """Extract FAIL_TO_PASS and PASS_TO_PASS success/failure counts from a report."""
    if not isinstance(report, dict):
        return {}
    instance_data = next(iter(report.values()), None)
    if not isinstance(instance_data, dict):
        return {}
    ts = instance_data.get("tests_status", {})
    result = {}
    for key in ["FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"]:
        section = ts.get(key, {})
        result[f"{key}_success"] = len(section.get("success", []))
        result[f"{key}_failure"] = len(section.get("failure", []))
        result[f"{key}_failed_tests"] = sorted(section.get("failure", []))
    return result


def _count_patch_lines(patch_text: str) -> dict:
    """Count added/removed/total lines in a patch."""
    if not patch_text:
        return {"added": 0, "removed": 0, "total": 0}
    added = 0
    removed = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"added": added, "removed": removed, "total": added + removed}


def _count_patch_files(patch_text: str) -> int:
    """Count number of files changed in a patch (diff --git headers)."""
    if not patch_text:
        return 0
    return sum(1 for line in patch_text.splitlines() if line.startswith("diff --git"))


def _get_repo(instance_id: str) -> str:
    """Extract repository name from instance_id (e.g., 'django__django-10999' -> 'django')."""
    return instance_id.split("__")[0] if "__" in instance_id else "unknown"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_convergence_to_incorrect_patches(force: bool = False):
    """
    Characterize bugs where convergence misleads.
    Analyzes all models in cfg.WITHIN_MODEL_MODELS.
    """
    output_jsonl = cfg.ANALYSIS_DIR / "convergence_to_incorrect_patches.jsonl"
    output_summary = cfg.ANALYSIS_DIR / "convergence_to_incorrect_patches_summary.txt"
    output_repo_csv = cfg.ANALYSIS_DIR / "convergence_to_incorrect_patches_repos.csv"

    if not force and output_jsonl.exists() and output_jsonl.stat().st_size > 0:
        print(f"[skip] Results already exist at {output_jsonl}")
        print(f"      Re-run with --force to recompute")
        return

    print("=" * 70)
    print("Convergence to Incorrect Patches")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Identify incorrectly converged bugs from convergence data
    # -----------------------------------------------------------------------
    converged_on_incorrect_patches = []  # 6/6 convergence, resolved_all=false
    true_converged = []   # 6/6 convergence, resolved_all=true
    partial_converged = []  # 6/6 convergence, resolved_any=true but resolved_all=false

    for method in ["syntactic", "semantic"]:
        jsonl_path = cfg.CONVERGENCE_JSONL[method]
        if not jsonl_path.exists():
            print(f"[warn] {jsonl_path} not found, skipping")
            continue

        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["model"] not in cfg.WITHIN_MODEL_MODELS:
                    continue
                if rec["bucket"] != "6/6":
                    continue

                entry = {
                    "instance_id": rec["instance_id"],
                    "model": rec["model"],
                    "method": method,
                    "n_runs": rec["n_runs"],
                    "n_resolved": rec["n_resolved"],
                    "resolved_any": rec["resolved_any"],
                    "resolved_all": rec["resolved_all"],
                }

                if rec["resolved_all"]:
                    true_converged.append(entry)
                elif rec["resolved_any"]:
                    partial_converged.append(entry)
                else:
                    converged_on_incorrect_patches.append(entry)

    for model in cfg.WITHIN_MODEL_MODELS:
        tc = [x for x in true_converged if x["model"] == model]
        pc = [x for x in partial_converged if x["model"] == model]
        fc = [x for x in converged_on_incorrect_patches if x["model"] == model]
        print(f"\n[info] {model} at 6/6 convergence:")
        print(f"       Resolved (all runs):    {len(tc)} (syntactic: "
              f"{sum(1 for x in tc if x['method']=='syntactic')}, "
              f"semantic: {sum(1 for x in tc if x['method']=='semantic')})")
        print(f"       Partial (some runs):    {len(pc)}")
        print(f"       Unresolved (no runs):   {len(fc)}")

    # -----------------------------------------------------------------------
    # Step 2: For each converged incorrectly patched bug, load patch + all run reports
    # -----------------------------------------------------------------------
    # Keep syntactic and semantic separate — keyed by (instance_id, model, method)
    seen_keys = set()
    all_problem_bugs = converged_on_incorrect_patches + partial_converged

    detailed_records = []
    for bug in all_problem_bugs:
        inst_id = bug["instance_id"]
        model = bug["model"]
        method = bug["method"]
        key = (inst_id, model, method)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        repo = _get_repo(inst_id)
        bug_record = {
            "instance_id": inst_id,
            "model": model,
            "method": method,
            "repo": repo,
            "category": "unresolved" if bug in converged_on_incorrect_patches else "partial",
            "n_runs": bug["n_runs"],
            "n_resolved": bug["n_resolved"],
        }

        # Load patch from first run
        ref_patch = io.load_patch(1, model, inst_id)
        patch_lines = _count_patch_lines(ref_patch)
        patch_files = _count_patch_files(ref_patch)
        bug_record["patch_added_lines"] = patch_lines["added"]
        bug_record["patch_removed_lines"] = patch_lines["removed"]
        bug_record["patch_total_lines"] = patch_lines["total"]
        bug_record["patch_files_changed"] = patch_files

        # Load reports from all runs
        run_reports = []
        total_fail_to_pass_success = 0
        total_fail_to_pass_failure = 0
        total_pass_to_pass_success = 0
        total_pass_to_pass_failure = 0
        all_failed_tests = []

        for run_id in cfg.RUNS:
            report = io.load_report(run_id, model, inst_id)
            if report is None:
                continue
            summary = _extract_test_summary(report)
            run_reports.append(summary)
            total_fail_to_pass_success += summary.get("FAIL_TO_PASS_success", 0)
            total_fail_to_pass_failure += summary.get("FAIL_TO_PASS_failure", 0)
            total_pass_to_pass_success += summary.get("PASS_TO_PASS_success", 0)
            total_pass_to_pass_failure += summary.get("PASS_TO_PASS_failure", 0)
            all_failed_tests.extend(summary.get("FAIL_TO_PASS_failed_tests", []))

        bug_record["total_fail_to_pass_success"] = total_fail_to_pass_success
        bug_record["total_fail_to_pass_failure"] = total_fail_to_pass_failure
        bug_record["total_pass_to_pass_success"] = total_pass_to_pass_success
        bug_record["total_pass_to_pass_failure"] = total_pass_to_pass_failure
        bug_record["total_tests"] = (
            total_fail_to_pass_success + total_fail_to_pass_failure
            + total_pass_to_pass_success + total_pass_to_pass_failure
        )
        bug_record["unique_failed_tests"] = sorted(set(all_failed_tests))
        bug_record["n_unique_failed_tests"] = len(set(all_failed_tests))

        # Classify failure pattern
        if total_fail_to_pass_failure > 0 and total_pass_to_pass_failure > 0:
            bug_record["failure_pattern"] = "ftp_regression"  # FAIL_TO_PASS gaps + PASS_TO_PASS regressions
        elif total_fail_to_pass_failure > 0:
            bug_record["failure_pattern"] = "ftp_incomplete"  # Some FAIL_TO_PASS tests still failing
        elif total_pass_to_pass_failure > 0:
            bug_record["failure_pattern"] = "regression"  # PASS_TO_PASS regressions
        else:
            bug_record["failure_pattern"] = "other"

        detailed_records.append(bug_record)

    # -----------------------------------------------------------------------
    # Step 3: Write detailed JSONL
    # -----------------------------------------------------------------------
    io.save_jsonl(detailed_records, output_jsonl)
    print(f"\n[ok] Saved {len(detailed_records)} records -> {output_jsonl}")

    # -----------------------------------------------------------------------
    # Step 4: Generate summary text for paper
    # -----------------------------------------------------------------------
    lines = []
    lines.append("Convergence to Incorrect Patches — Summary")
    lines.append("=" * 60)
    lines.append("")

    # Per-model, per-method overview
    for model in cfg.WITHIN_MODEL_MODELS:
        for method in ["syntactic", "semantic"]:
            method_records = [r for r in detailed_records if r["model"] == model and r["method"] == method]
            total_6_6 = sum(1 for x in true_converged if x["model"] == model and x["method"] == method) + len(method_records)
            if total_6_6 == 0:
                lines.append(f"{model} {method}: no 6/6 convergence")
                lines.append("")
                continue
            n_unresolved = sum(1 for r in method_records if r["category"] == "unresolved")
            n_partial = sum(1 for r in method_records if r["category"] == "partial")
            false_rate = len(method_records) / total_6_6 * 100 if total_6_6 > 0 else 0
            lines.append(f"{model} {method}: {len(method_records)}/{total_6_6} converged to incorrect patches ({false_rate:.1f}%)")
            lines.append(f"  - Unresolved in any run: {n_unresolved}")
            lines.append(f"  - Resolved in some but not all runs: {n_partial}")

            pattern_counts = Counter(r["failure_pattern"] for r in method_records)
            if pattern_counts:
                lines.append(f"  Failure patterns:")
                for pattern, count in pattern_counts.most_common():
                    label = {
                        "ftp_incomplete": "Incomplete fixes (FAIL_TO_PASS tests still failing)",
                        "ftp_regression": "Incomplete fixes + regressions",
                        "regression": "Regressions only (PASS_TO_PASS failures)",
                        "other": "Other",
                    }.get(pattern, pattern)
                    lines.append(f"    {label}: {count}")
            lines.append("")

    # Aggregate overview per method
    for method in ["syntactic", "semantic"]:
        m_records = [r for r in detailed_records if r["method"] == method]
        m_total_6_6 = sum(1 for x in true_converged if x["method"] == method) + len(m_records)
        if m_total_6_6 > 0:
            lines.append(f"Total {method}: {len(m_records)}/{m_total_6_6} converged to incorrect patches ({len(m_records)/m_total_6_6*100:.1f}%)")
            m_incomplete = sum(1 for r in m_records if r["failure_pattern"] in ("ftp_incomplete", "ftp_regression"))
            lines.append(f"  Incomplete fixes: {m_incomplete}/{len(m_records)} ({m_incomplete/len(m_records)*100:.1f}%)" if m_records else "")
    lines.append("")

    # Per-tier summary: 500 bugs → converged 6/6 → incorrect → failure modes
    lines.append("Per-tier summary (each model sees 500 bugs)")
    lines.append("=" * 60)
    for tier_name in ["large", "mid", "small"]:
        tier_models = cfg.MODEL_TIERS[tier_name]
        lines.append(f"\n--- {tier_name.upper()} tier ({', '.join(tier_models)}) ---")
        for method in ["syntactic", "semantic"]:
            lines.append(f"  {method}:")
            for model in tier_models:
                m_true = [x for x in true_converged if x["model"] == model and x["method"] == method]
                m_false = [r for r in detailed_records if r["model"] == model and r["method"] == method]
                total_6_6 = len(m_true) + len(m_false)
                n_incorrect = len(m_false)
                pattern_counts = Counter(r["failure_pattern"] for r in m_false)
                n_incomplete = pattern_counts.get("ftp_incomplete", 0) + pattern_counts.get("ftp_regression", 0)
                n_regression = pattern_counts.get("regression", 0)
                n_other = pattern_counts.get("other", 0)
                false_rate = n_incorrect / total_6_6 * 100 if total_6_6 > 0 else 0
                lines.append(
                    f"    {model}: {total_6_6}/500 converged, "
                    f"{n_incorrect} incorrect ({false_rate:.1f}%) "
                    f"[incomplete={n_incomplete}, regression={n_regression}, other={n_other}]"
                )
            # Tier aggregate
            tier_true = [x for x in true_converged if x["model"] in tier_models and x["method"] == method]
            tier_false = [r for r in detailed_records if r["model"] in tier_models and r["method"] == method]
            tier_total_6_6 = len(tier_true) + len(tier_false)
            tier_incorrect = len(tier_false)
            tier_patterns = Counter(r["failure_pattern"] for r in tier_false)
            tier_incomplete = tier_patterns.get("ftp_incomplete", 0) + tier_patterns.get("ftp_regression", 0)
            tier_regression = tier_patterns.get("regression", 0)
            tier_other = tier_patterns.get("other", 0)
            tier_false_rate = tier_incorrect / tier_total_6_6 * 100 if tier_total_6_6 > 0 else 0
            n_models = len(tier_models)
            lines.append(
                f"    TIER TOTAL ({n_models} models × 500 bugs = {n_models * 500}): "
                f"{tier_total_6_6} converged, {tier_incorrect} incorrect ({tier_false_rate:.1f}%) "
                f"[incomplete={tier_incomplete}, regression={tier_regression}, other={tier_other}]"
            )
    lines.append("")

    # Repository distribution
    repo_counts = Counter(r["repo"] for r in detailed_records)
    lines.append("Repository distribution (all models):")
    for repo, count in repo_counts.most_common():
        lines.append(f"  {repo}: {count}")
    lines.append("")

    # Patch size statistics
    if detailed_records:
        avg_files = sum(r["patch_files_changed"] for r in detailed_records) / len(detailed_records)
        avg_lines = sum(r["patch_total_lines"] for r in detailed_records) / len(detailed_records)
        lines.append(f"Average patch size: {avg_files:.1f} files, {avg_lines:.0f} lines")
        lines.append("")

    # Per-bug details
    lines.append("Per-bug details:")
    lines.append("-" * 60)
    for r in sorted(detailed_records, key=lambda x: (x["model"], x["method"], x["category"], x["repo"], x["instance_id"])):
        lines.append(f"\n  {r['instance_id']} [{r['model']}] [{r['method']}] [{r['category']}]")
        lines.append(f"    Repo: {r['repo']}")
        lines.append(f"    Patch: {r['patch_files_changed']} files, "
                      f"+{r['patch_added_lines']}/-{r['patch_removed_lines']} lines")
        lines.append(f"    Runs resolved: {r['n_resolved']}/{r['n_runs']}")
        lines.append(f"    Failure pattern: {r['failure_pattern']}")
        lines.append(f"    FAIL_TO_PASS: {r['total_fail_to_pass_success']} pass, "
                      f"{r['total_fail_to_pass_failure']} fail (across all runs)")
        lines.append(f"    PASS_TO_PASS: {r['total_pass_to_pass_success']} pass, "
                      f"{r['total_pass_to_pass_failure']} fail (across all runs)")
        if r["n_unique_failed_tests"] > 0:
            lines.append(f"    Unique failed tests: {r['n_unique_failed_tests']}")
            for t in r["unique_failed_tests"][:5]:
                lines.append(f"      - {t}")
            if r["n_unique_failed_tests"] > 5:
                lines.append(f"      ... and {r['n_unique_failed_tests'] - 5} more")

    summary_text = "\n".join(lines)
    output_summary.write_text(summary_text, encoding="utf-8")
    print(f"[ok] Saved summary -> {output_summary}")

    # -----------------------------------------------------------------------
    # Step 5: Write repo distribution CSV
    # -----------------------------------------------------------------------
    csv_lines = ["repo,count,category"]
    for r in sorted(detailed_records, key=lambda x: x["repo"]):
        csv_lines.append(f"{r['repo']},{r['instance_id']},{r['category']}")
    # Also add repo summary
    csv_lines.append("")
    csv_lines.append("repo,total_unresolved")
    for repo, count in repo_counts.most_common():
        csv_lines.append(f"{repo},{count}")

    repo_csv_text = "\n".join(csv_lines)
    output_repo_csv.write_text(repo_csv_text, encoding="utf-8")
    print(f"[ok] Saved repo distribution -> {output_repo_csv}")

    # -----------------------------------------------------------------------
    # Step 6: Print summary to stdout
    # -----------------------------------------------------------------------
    print("\n" + summary_text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    force = "--force" in sys.argv
    analyze_convergence_to_incorrect_patches(force=force)
