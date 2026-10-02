#!/usr/bin/env python3
"""
Convergence vs Problem Difficulty

Uses the official SWE-bench Verified difficulty labels from HuggingFace:
  - "<15 min fix"  (Easy)
  - "15 min - 1 hour" (Medium)
  - "1-4 hours" (Hard)
  - ">4 hours" (Very Hard)

Cross-tabulates difficulty tier with convergence bucket and resolution rate
for each model, producing the data needed for a stratified analysis figure.

Prerequisite: difficulty_labels.csv in the scripts directory (instance_id -> difficulty).
If not present, it will be fetched from HuggingFace.

Output:
- difficulty_crosstab_<model>_<method>.csv
- difficulty_paper_table.csv
- difficulty_summary.txt
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import analysis_config as cfg
import analysis_data_io as io


# ---------------------------------------------------------------------------
# Difficulty labels
# ---------------------------------------------------------------------------

DIFFICULTY_LABELS = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
DIFFICULTY_SHORT = {
    "<15 min fix": "Easy",
    "15 min - 1 hour": "Medium",
    "1-4 hours": "Hard",
    ">4 hours": "Very Hard",
}


def load_difficulty_labels() -> dict:
    """Load instance_id -> difficulty from CSV. Fetches from HF if not present."""
    csv_path = Path(__file__).parent / "difficulty_labels.csv"

    if csv_path.exists():
        labels = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["instance_id"]] = row["difficulty"]
        print(f"[info] Loaded {len(labels)} difficulty labels from {csv_path}")
        return labels

    # Fallback: fetch from HuggingFace
    print("[info] difficulty_labels.csv not found, fetching from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        labels = {ex["instance_id"]: ex["difficulty"] for ex in ds}
        # Save for future use
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "difficulty"])
            for inst_id, diff in sorted(labels.items()):
                writer.writerow([inst_id, diff])
        print(f"[info] Saved {len(labels)} labels to {csv_path}")
        return labels
    except ImportError:
        print("[error] 'datasets' library not installed. Install with: pip install datasets")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_difficulty(force: bool = False):
    """Run difficulty analysis for all models."""
    labels = load_difficulty_labels()
    buckets = cfg.CONVERGENCE_BUCKETS  # ["6/6", "5/6", ..., "1/6"]

    summary_lines = []
    summary_lines.append("Convergence vs Problem Difficulty")
    summary_lines.append("=" * 60)
    summary_lines.append("")

    tier_order = ["large", "mid", "small"]
    tier_display = {"large": "Large-tier LLMs", "mid": "Mid-tier LLMs", "small": "Small-tier LLMs"}

    for method in ["semantic", "syntactic"]:
        summary_lines.append(f"{'=' * 60}")
        summary_lines.append(f"Method: {method}")
        summary_lines.append(f"{'=' * 60}")
        summary_lines.append("")

        jsonl_path = cfg.CONVERGENCE_JSONL[method]
        if not jsonl_path.exists():
            print(f"[warn] {jsonl_path} not found, skipping {method}")
            continue

        # Load all convergence records
        all_convergence = []
        with open(jsonl_path) as f:
            for line in f:
                all_convergence.append(json.loads(line))

        # Per-model cross-tab (for per-model CSV output)
        for model_name in cfg.WITHIN_MODEL_MODELS:
            convergence_data = {r["instance_id"]: r for r in all_convergence if r["model"] == model_name}

            table = defaultdict(lambda: defaultdict(int))
            resolved_table = defaultdict(lambda: defaultdict(int))
            for inst_id, conv in convergence_data.items():
                diff = labels.get(inst_id, "unknown")
                bucket = conv["bucket"]
                table[diff][bucket] += 1
                if conv["resolved_all"]:
                    resolved_table[diff][bucket] += 1

            safe_name = model_name.replace("-", "_").replace(".", "_")
            csv_path = cfg.ANALYSIS_DIR / f"difficulty_difficulty_crosstab_{safe_name}_{method}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "method", "difficulty", "difficulty_short", "bucket",
                    "total", "resolved_all", "precision_all"
                ])
                for diff in DIFFICULTY_LABELS:
                    for b in buckets:
                        n = table[diff][b]
                        r = resolved_table[diff][b]
                        pct = round(r / n, 4) if n > 0 else ""
                        writer.writerow([
                            model_name, method, diff, DIFFICULTY_SHORT[diff], b,
                            n, r, pct
                        ])
            print(f"[ok] Saved -> {csv_path}")

        # Per-tier aggregated cross-tab (for paper table)
        for tier in tier_order:
            tier_models = cfg.MODEL_TIERS[tier]
            print(f"\n--- {tier_display[tier]} ({method}) ---")
            summary_lines.append(f"--- {tier_display[tier]} ({method}) ---")
            summary_lines.append("")

            tier_table = defaultdict(lambda: defaultdict(int))
            tier_resolved = defaultdict(lambda: defaultdict(int))

            for r in all_convergence:
                if r["model"] not in tier_models:
                    continue
                diff = labels.get(r["instance_id"], "unknown")
                bucket = r["bucket"]
                tier_table[diff][bucket] += 1
                if r["resolved_all"]:
                    tier_resolved[diff][bucket] += 1

            # Save tier CSV
            safe_tier = tier.replace("-", "_")
            csv_path = cfg.ANALYSIS_DIR / f"difficulty_difficulty_crosstab_tier_{safe_tier}_{method}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tier", "method", "difficulty", "difficulty_short", "bucket",
                    "total", "resolved_all", "precision_all"
                ])
                for diff in DIFFICULTY_LABELS:
                    for b in buckets:
                        n = tier_table[diff][b]
                        r = tier_resolved[diff][b]
                        pct = round(r / n, 4) if n > 0 else ""
                        writer.writerow([
                            tier, method, diff, DIFFICULTY_SHORT[diff], b,
                            n, r, pct
                        ])
            print(f"[ok] Saved -> {csv_path}")

            for diff in DIFFICULTY_LABELS:
                total = sum(tier_table[diff].values())
                total_resolved = sum(tier_resolved[diff].values())
                if total == 0:
                    continue
                pct = total_resolved / total * 100
                short = DIFFICULTY_SHORT[diff]
                line = f"  {short:<12} (n={total:>3}, resolved: {total_resolved:>3}/{total} = {pct:.1f}%)"
                print(line)
                summary_lines.append(line)
                for b in buckets:
                    n = tier_table[diff][b]
                    r = tier_resolved[diff][b]
                    if n == 0:
                        continue
                    pct_b = r / n * 100
                    sub = f"    {b}: {n:>3} bugs, {r:>3} resolved ({pct_b:.1f}%)"
                    print(sub)
                    summary_lines.append(sub)
                summary_lines.append("")

    # Save summary
    summary_path = cfg.ANALYSIS_DIR / "difficulty_summary.txt"
    summary_text = "\n".join(summary_lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n[ok] Saved summary -> {summary_path}")

    # Print full summary
    print("\n" + summary_text)

    # -----------------------------------------------------------------
    # Paper table: Baseline + 6/6 correctness per tier x difficulty
    # -----------------------------------------------------------------
    generate_paper_table(labels)


def generate_paper_table(labels: dict):
    """Produce the paper difficulty table: Baseline vs 6/6 correctness.

    Baseline = mean(n_resolved / 6) across all (model, bug) pairs in a
    tier x difficulty group.  This is method-independent (resolution does
    not depend on the convergence method), so we compute it once from
    the semantic JSONL (arbitrary choice, numbers identical either way).

    6/6 correctness = resolved_all rate among bugs that converge at 6/6,
    computed separately for semantic and syntactic.
    """
    tier_order = ["large", "mid", "small"]
    tier_display = {"large": "Large", "mid": "Mid", "small": "Small"}

    # Load convergence records for both methods
    records = {}
    for method in ["semantic", "syntactic"]:
        jsonl_path = cfg.CONVERGENCE_JSONL[method]
        if not jsonl_path.exists():
            print(f"[warn] {jsonl_path} not found, skipping paper table for {method}")
            continue
        recs = []
        with open(jsonl_path) as f:
            for line in f:
                recs.append(json.loads(line))
        records[method] = recs

    if not records:
        return

    # --- Baseline (method-independent) ---
    # Use semantic records (n_resolved is the same regardless of method)
    base_method = "semantic" if "semantic" in records else "syntactic"
    # Accumulate sum(n_resolved/6) and count per tier x difficulty
    baseline_sum = defaultdict(lambda: defaultdict(float))
    baseline_count = defaultdict(lambda: defaultdict(int))
    for r in records[base_method]:
        tier = cfg.MODEL_TO_TIER.get(r["model"])
        if tier is None:
            continue
        diff = labels.get(r["instance_id"], "unknown")
        if diff == "unknown":
            continue
        baseline_sum[tier][diff] += r["n_resolved"] / 6.0
        baseline_count[tier][diff] += 1

    # --- 6/6 correctness per method ---
    six_six = {}
    for method, recs in records.items():
        # Filter to 6/6 bucket, group by tier x difficulty
        total = defaultdict(lambda: defaultdict(int))
        resolved = defaultdict(lambda: defaultdict(int))
        for r in recs:
            if r["bucket"] != "6/6":
                continue
            tier = cfg.MODEL_TO_TIER.get(r["model"])
            if tier is None:
                continue
            diff = labels.get(r["instance_id"], "unknown")
            if diff == "unknown":
                continue
            total[tier][diff] += 1
            if r["resolved_all"]:
                resolved[tier][diff] += 1
        six_six[method] = (total, resolved)

    # --- Output ---
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("PAPER TABLE: Baseline vs 6/6 Correctness per Tier x Difficulty")
    lines.append("=" * 70)
    lines.append("")

    # CSV output
    csv_path = cfg.ANALYSIS_DIR / "difficulty_paper_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["tier", "difficulty", "difficulty_short",
                  "baseline_n", "baseline_rate",
                  "semantic_6_6_total", "semantic_6_6_resolved", "semantic_6_6_rate",
                  "syntactic_6_6_total", "syntactic_6_6_resolved", "syntactic_6_6_rate"]
        writer.writerow(header)

        for tier in tier_order:
            tier_label = tier_display[tier]
            lines.append(f"--- {tier_label}-tier LLMs ---")

            for diff in DIFFICULTY_LABELS:
                short = DIFFICULTY_SHORT[diff]
                # Baseline
                n_base = baseline_count[tier][diff]
                rate_base = baseline_sum[tier][diff] / n_base * 100 if n_base > 0 else 0
                line = f"  {short:<12}  Baseline: {rate_base:5.1f}% (n={n_base})"

                row = [tier, diff, short, n_base, round(rate_base, 1)]

                for method in ["semantic", "syntactic"]:
                    if method in six_six:
                        t_total, t_resolved = six_six[method]
                        n = t_total[tier][diff]
                        r = t_resolved[tier][diff]
                        pct = r / n * 100 if n > 0 else 0
                        line += f"  |  {method} 6/6: {pct:5.1f}% ({r}/{n})"
                        row.extend([n, r, round(pct, 1)])
                    else:
                        row.extend(["", "", ""])

                lines.append(line)
                writer.writerow(row)

            lines.append("")

    print("\n".join(lines))
    print(f"\n[ok] Saved paper table -> {csv_path}")

    # Append to summary file
    summary_path = cfg.ANALYSIS_DIR / "difficulty_summary.txt"
    with open(summary_path, "a") as f:
        f.write("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    force = "--force" in sys.argv
    analyze_difficulty(force=force)
