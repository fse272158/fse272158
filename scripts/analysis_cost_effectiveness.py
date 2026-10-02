#!/usr/bin/env python3
"""
Cost-effectiveness analysis for within-model and cross-model convergence.

Motivation: we ask, if a practitioner runs N times and all N agree, how likely
is the patch to be correct? This lets a practitioner decide how much budget
to spend based on how much confidence they need.

Within-model (Figure 2):
    For each N in 2..6, for each bug, take all C(6, N) subsets of runs.
    Keep only subsets where all N runs converge (syntactically or
    semantically).
    For each convergent subset, compute resolved_runs / N.  Average these
    fractions across all convergent subsets for this bug.  Boxplot these
    per-bug averages across bugs, for each N.

    A bug where all 6 runs converge contributes C(6,2)=15 pairs to N=2,
    C(6,3)=20 triples to N=3, etc., so higher-N convergence "feeds into"
    the lower-N boxplots.

    We also report how many bugs contribute to each boxplot (bugs that have
    at least one convergent subset at that N), since for high N very few
    bugs may have N runs that all converge.

Cross-model (Figure 3):
    Same idea, but N is about how many runs from each model agree with each
    other. For a given N, take a subset of N runs from model A and a subset
    of N runs from model B. The cross-subset "converges" if all N patches
    from A match all N patches from B (i.e. all N x N pairs agree). For each
    convergent cross-model subset, compute resolved_runs / (2*N), then
    average across subsets per bug and boxplot across bugs.

Output:
    analysis/cost_effectiveness_within_model.jsonl
        One row per (model, method, N, bug):
        {instance_id, model, method, n, n_convergent_subsets,
         n_correct_subsets, correctness_ratio}
    analysis/cost_effectiveness_cross_model.jsonl
        One row per (pair, method, N, bug):
        {instance_id, pair, method, n, n_convergent_subsets,
         n_correct_subsets, correctness_ratio}
    analysis/cost_effectiveness_summary.json
        Boxplot stats (median, q1, q3, whiskers, mean, count) per
        (model/pair, method, N), ready for plotting.
    analysis/paper_cost_effectiveness_within_model.png
    analysis/paper_cost_effectiveness_cross_model.png
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import analysis_config as cfg
import analysis_data_io as io
import analysis_similarity as sim

# ---------------------------------------------------------------------------
# In-memory data cache
# ---------------------------------------------------------------------------

_cache = {}  # (model, run_id, inst_id) -> (patch, report)
_cache_instance_ids = {}  # model -> set of inst_ids

_log_file = cfg.ANALYSIS_DIR / "cost_effectiveness_progress.log"


def _log(msg):
    from datetime import datetime
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(_log_file, "a") as f:
        f.write(line + "\n")


def _load_model_data(model: str):
    """Load all patches and reports for a model into the cache (once)."""
    if model in _cache_instance_ids:
        return
    _log(f"[cache] {model} loading...")
    inst_ids = set()
    for run_id in cfg.RUNS:
        inst_ids.update(io.load_all_instance_ids(run_id, model))
    for inst_id in inst_ids:
        for run_id in cfg.RUNS:
            _cache[(model, run_id, inst_id)] = (
                io.load_patch(run_id, model, inst_id),
                io.load_report(run_id, model, inst_id),
            )
    _cache_instance_ids[model] = inst_ids
    _log(f"[cache] {model} done: {len(inst_ids)} bugs x {len(cfg.RUNS)} runs")


def _get(model, run_id, inst_id):
    return _cache.get((model, run_id, inst_id), (None, None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_resolved(report: dict) -> bool:
    """Check if a report indicates the instance was resolved."""
    if not isinstance(report, dict):
        return False
    instance_data = next(iter(report.values()), None)
    if not isinstance(instance_data, dict):
        return False
    return bool(instance_data.get("resolved", False))


def _all_converge(subset_patches, subset_reports, method) -> bool:
    """Return True if all patches in the subset match each other under `method`."""
    if len(subset_patches) < 2:
        return True
    ref_patch, ref_report = subset_patches[0], subset_reports[0]
    for patch, report in zip(subset_patches[1:], subset_reports[1:]):
        if not sim.compare_patches(ref_patch, patch, ref_report, report, method):
            return False
    return True


def _subset_is_correct(subset_reports) -> bool:
    """A convergent subset is 'correct' if any of its runs resolved the bug.

    Rationale: if the model converged on a patch and at least one run that
    produced this patch resolved the bug, the converged-on patch is correct.
    """
    for report in subset_reports:
        if _is_resolved(report):
            return True
    return False


# ---------------------------------------------------------------------------
# Within-model
# ---------------------------------------------------------------------------

def compute_within_model(model: str, method: str) -> list[dict]:
    """Compute cost-effectiveness data for one model x method.

    Returns one dict per N (2..6) with boxplot stats ready for ax.bxp():
        {model, method, n, median, q1, q3, whisker_min, whisker_max, mean, count}
    """
    _load_model_data(model)
    all_instance_ids = _cache_instance_ids[model]

    results = []
    for n in range(2, 7):
        ratios = []
        for inst_id in sorted(all_instance_ids):
            runs_with_data = []
            for run_id in cfg.RUNS:
                patch, report = _get(model, run_id, inst_id)
                if patch is not None:
                    runs_with_data.append(run_id)

            if len(runs_with_data) < 2:
                continue

            convergent_subsets = 0
            total_correct_runs = 0
            total_runs_in_subsets = 0

            for subset in combinations(runs_with_data, n):
                patches = [_get(model, r, inst_id)[0] for r in subset]
                reports = [_get(model, r, inst_id)[1] for r in subset]

                if _all_converge(patches, reports, method):
                    convergent_subsets += 1
                    resolved_count = sum(1 for r in reports if _is_resolved(r))
                    total_correct_runs += resolved_count
                    total_runs_in_subsets += n

            if convergent_subsets > 0:
                ratio = total_correct_runs / total_runs_in_subsets
                ratios.append(round(ratio, 4))

        stats = _boxplot_stats(ratios)
        stats["ratios"] = ratios
        results.append({
            "model": model,
            "method": method,
            "n": n,
            **stats,
        })

    return results


# ---------------------------------------------------------------------------
# Cross-model
# ---------------------------------------------------------------------------

def compute_cross_model(model_a: str, model_b: str, method: str) -> list[dict]:
    """Compute cost-effectiveness data for a cross-model pair x method.

    For a given N, take a subset of N runs from A and N runs from B. The
    cross-subset "converges" if all N patches from A match all N patches from
    B (i.e. all N x N pairs agree). For each convergent cross-model subset,
    correctness is resolved_runs / (2*N), averaged across subsets per bug.

    Returns one dict per N (2..6) with boxplot stats ready for ax.bxp():
        {pair, model_a, model_b, method, n, median, q1, q3, whisker_min,
         whisker_max, mean, count}
    """
    _load_model_data(model_a)
    _load_model_data(model_b)
    all_instance_ids = _cache_instance_ids[model_a] | _cache_instance_ids[model_b]

    pair_key = f"{model_a}_vs_{model_b}"
    results = []

    for n in range(2, 7):
        ratios = []
        for inst_id in sorted(all_instance_ids):
            runs_a = [r for r in cfg.RUNS
                      if _get(model_a, r, inst_id)[0] is not None]
            runs_b = [r for r in cfg.RUNS
                      if _get(model_b, r, inst_id)[0] is not None]

            if len(runs_a) < 2 or len(runs_b) < 2:
                continue
            if len(runs_a) < n or len(runs_b) < n:
                continue

            convergent_subsets = 0
            total_correct_runs = 0
            total_runs_in_subsets = 0

            for sub_a in combinations(runs_a, n):
                patches_a = [_get(model_a, r, inst_id)[0] for r in sub_a]
                reports_a = [_get(model_a, r, inst_id)[1] for r in sub_a]

                for sub_b in combinations(runs_b, n):
                    patches_b = [_get(model_b, r, inst_id)[0] for r in sub_b]
                    reports_b = [_get(model_b, r, inst_id)[1] for r in sub_b]

                    all_patches = patches_a + patches_b
                    all_reports = reports_a + reports_b

                    if _all_converge(all_patches, all_reports, method):
                        convergent_subsets += 1
                        resolved_count = sum(1 for r in all_reports if _is_resolved(r))
                        total_correct_runs += resolved_count
                        total_runs_in_subsets += 2 * n

            if convergent_subsets > 0:
                ratio = total_correct_runs / total_runs_in_subsets
                ratios.append(round(ratio, 4))

        stats = _boxplot_stats(ratios)
        stats["ratios"] = ratios
        results.append({
            "pair": pair_key,
            "model_a": model_a,
            "model_b": model_b,
            "method": method,
            "n": n,
            **stats,
        })

    return results


# ---------------------------------------------------------------------------
# Boxplot stats
# ---------------------------------------------------------------------------

def _boxplot_stats(values: list[float]) -> dict:
    """Compute boxplot statistics for a list of values."""
    if not values:
        return {"median": None, "q1": None, "q3": None,
                "whisker_min": None, "whisker_max": None,
                "mean": None, "count": 0}

    s = sorted(values)
    n = len(s)

    def quantile(q):
        if n == 1:
            return s[0]
        idx = (n - 1) * q
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    q1 = quantile(0.25)
    q3 = quantile(0.75)
    median = quantile(0.5)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    whisker_min = min(v for v in s if v >= lower_fence)
    whisker_max = max(v for v in s if v <= upper_fence)

    fliers = [round(v, 4) for v in s if v < lower_fence or v > upper_fence]

    return {
        "median": round(median, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "whisker_min": round(whisker_min, 4),
        "whisker_max": round(whisker_max, 4),
        "mean": round(sum(s) / n, 4),
        "count": n,
        "fliers": fliers,
    }


def build_summary_key(record: dict) -> str:
    """Build the summary key for a record (within or cross)."""
    if "pair" in record:
        return f"{record['pair']}/{record['method']}/n={record['n']}"
    return f"{record['model']}/{record['method']}/n={record['n']}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_METHODS = ["semantic", "syntactic"]

METHOD_LABELS = {
    "syntactic": "Syntactic comparison",
    "semantic": "Semantic comparison",
}


def _plot_boxplots(records: list[dict], group_key: str, output_path: Path,
                   title: str, group_order: list[str] | None = None) -> None:
    """Boxplot figure matching the style of analysis_plot.py.

    Two panels (syntactic, semantic). Each panel shows all models/pairs side
    by side, x-axis = N (2..6), y = correctness ratio. Reads from the JSONL
    format where each record is one row per (model/pair, method, N) with
    boxplot stats and raw per-bug ratios.

    Uses ax.bxp() with pre-computed stats plus the fliers list computed from
    the raw ratios.  MIN_BOX_HEIGHT padding ensures collapsed boxes (Q1=Q3)
    remain visible.  The mean is shown as the median line for visibility.

    Reuses _setup_style() from analysis_plot so all figures share font sizes,
    grid, spine, and save settings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import analysis_plot  # reuse _setup_style for consistent paper style
    analysis_plot._setup_style()

    # Index: (method, group_label, n) -> record (stats + raw ratios)
    parsed = {}
    all_groups = set()
    for r in records:
        label = r.get(group_key, "")
        key = (r["method"], label, r["n"])
        parsed[key] = r
        all_groups.add(label)

    if group_order:
        ordered = [g for g in group_order if g in all_groups]
        remaining = sorted(g for g in all_groups if g not in ordered)
        all_groups = ordered + remaining
    else:
        all_groups = sorted(all_groups)
    n_methods = len(PLOT_METHODS)
    n_values = [2, 3, 4, 5, 6]

    fig, axes = plt.subplots(n_methods, 1, figsize=(40, 6 * n_methods), sharey=True)
    fig.subplots_adjust(hspace=1.0)
    if n_methods == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab20")
    model_color_idx = {g: i for i, g in enumerate(all_groups)}

    # Build legend handles shared across panels (same models/pairs in each)
    legend_handles = []
    for group in all_groups:
        color_idx = model_color_idx.get(group, len(legend_handles))
        color = cmap(color_idx)
        display_label = group.replace("_vs_", " & ")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7,
                                          label=display_label))

    for ax_idx, method in enumerate(PLOT_METHODS):
        ax = axes[ax_idx]

        G = len(all_groups)
        width = min(0.35, 0.7 / G)
        for label_idx, group in enumerate(all_groups):
            # Build boxplot stats list for this group at each N
            box_data = []
            counts = []
            for n in n_values:
                rec = parsed.get((method, group, n))
                if rec and rec["count"] > 0:
                    box_data.append(rec)
                    counts.append(rec["count"])
                else:
                    box_data.append(None)
                    counts.append(0)

            color_idx = model_color_idx.get(group, label_idx)
            base_color = cmap(color_idx)

            # Offset positions so groups don't overlap within an N group
            offset = (label_idx - (G - 1) / 2) * width
            positions = [n + offset for n in n_values]

            # Filter out None entries for boxplot
            valid_data = [(pos, bd) for pos, bd in zip(positions, box_data) if bd is not None]
            if not valid_data:
                continue

            valid_positions, valid_stats = zip(*valid_data)

            # Build bxp stats with fliers from raw ratios.
            # Show mean where the median line would be (so it's visible
            # even when the box is at the top of the plot).
            MIN_BOX_HEIGHT = 0.02
            bxp_stats = []
            for bd in valid_stats:
                actual_q1 = bd["q1"]
                actual_q3 = bd["q3"]
                actual_mean = bd["mean"]
                actual_whislo = bd["whisker_min"]
                actual_whishi = bd["whisker_max"]
                fliers = bd.get("fliers", [])

                box_height = actual_q3 - actual_q1
                if box_height < MIN_BOX_HEIGHT:
                    padding = (MIN_BOX_HEIGHT - box_height) / 2
                    display_q1 = max(0.0, actual_q1 - padding)
                    display_q3 = min(1.0, actual_q3 + padding)
                else:
                    display_q1 = actual_q1
                    display_q3 = actual_q3

                bxp_stats.append({
                    "med": actual_mean,
                    "q1": display_q1,
                    "q3": display_q3,
                    "whislo": actual_whislo,
                    "whishi": actual_whishi,
                    "fliers": fliers,
                })
            bp = ax.bxp(bxp_stats, positions=valid_positions, widths=width,
                         patch_artist=True,
                         medianprops=dict(color="black", linewidth=2),
                         showfliers=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(base_color)
                patch.set_alpha(0.7)
            # Style fliers
            for flier in bp["fliers"]:
                flier.set(marker="o", markerfacecolor=base_color, markersize=5,
                          alpha=0.5, markeredgecolor="none")

            # Annotate bug counts above each box (stagger vertically to avoid overlap)
            for pos, cnt in zip(valid_positions, [bd["count"] for bd in valid_stats]):
                y_offset = 1.04 + (label_idx % 2) * 0.14
                ax.annotate(f"{cnt}",
                            (pos, y_offset), ha="center", fontsize=cfg.PLOT_FONT_COUNT_ANNOTATION, color="gray")

        ax.set_title(METHOD_LABELS[method], fontsize=cfg.PLOT_FONT_TITLE, pad=20)
        if ax is axes[-1]:
            if group_key == "model":
                ax.set_xlabel("N (Convergence)", fontsize=cfg.PLOT_FONT_AXIS_LABEL)
            else:
                ax.set_xlabel("N (Agreement)", fontsize=cfg.PLOT_FONT_AXIS_LABEL)
        ax.set_ylabel("Correctness ratio", fontsize=cfg.PLOT_FONT_AXIS_LABEL)

        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values], fontsize=cfg.PLOT_FONT_TICK)
        ax.tick_params(axis="y", labelsize=cfg.PLOT_FONT_TICK)
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylim(0, 1.20)
        ax.set_xlim(1.5, 6.5)

    ncol = max(1, (len(all_groups) + 2) // 3)
    fig.legend(handles=legend_handles, loc="center", framealpha=0.9,
               fontsize=cfg.PLOT_FONT_LEGEND, ncol=ncol,
               bbox_to_anchor=(0.5, 0.5))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_plots(within_records: list[dict], cross_records: list[dict]) -> None:
    """Generate both cost-effectiveness figures from the JSONL records."""
    within_out = cfg.ANALYSIS_DIR / "paper_cost_effectiveness_within_model.png"
    cross_out = cfg.ANALYSIS_DIR / "paper_cost_effectiveness_cross_model.png"

    within_order = cfg.WITHIN_MODEL_MODELS
    cross_order = [f"{a}_vs_{b}" for a, b in cfg.CROSS_MODEL_PAIRS]

    _plot_boxplots(within_records, "model", within_out,
                   "Within-Model Convergence Cost-Effectiveness",
                   group_order=within_order)
    _plot_boxplots(cross_records, "pair", cross_out,
                   "Cross-Model Agreement Cost-Effectiveness",
                   group_order=cross_order)
    print(f"[ok] Saved figures:\n  {within_out}\n  {cross_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cost-effectiveness analysis: if N runs all agree, "
                    "how likely is the patch to be correct?")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if output exists")
    parser.add_argument("--method", default="all",
                        choices=["syntactic", "semantic", "all"],
                        help="Similarity method (default: all)")
    args = parser.parse_args()

    _CE_METHODS = ["syntactic", "semantic"]
    methods = _CE_METHODS if args.method == "all" else [args.method]

    within_out = cfg.ANALYSIS_DIR / "cost_effectiveness_within_model.jsonl"
    cross_out = cfg.ANALYSIS_DIR / "cost_effectiveness_cross_model.jsonl"
    summary_out = cfg.ANALYSIS_DIR / "cost_effectiveness_summary.json"

    # Load existing if present and not forced
    existing_within = []
    existing_cross = []
    if not args.force and within_out.is_file():
        existing_within = io.load_jsonl(within_out)
    if not args.force and cross_out.is_file():
        existing_cross = io.load_jsonl(cross_out)

    if not args.force and within_out.is_file() and cross_out.is_file():
        make_plots(existing_within, existing_cross)
        print(f"[ok] Plots regenerated from {within_out.name} and {cross_out.name}")
        return

    if _log_file.exists():
        _log_file.unlink()
    _log("start")

    for model in cfg.WITHIN_MODEL_MODELS:
        _load_model_data(model)
    _log("all models cached")

    new_within = []
    new_cross = []

    # Within-model
    for model in cfg.WITHIN_MODEL_MODELS:
        for method in methods:
            key = f"{model}/{method}"
            if not args.force and any(
                r["model"] == model and r["method"] == method
                for r in existing_within
            ):
                _log(f"[skip] within {key}")
                continue
            _log(f"[run] within {key}")
            new_within.extend(compute_within_model(model, method))
            _log(f"[done] within {key}")

    # Cross-model
    for model_a, model_b in cfg.CROSS_MODEL_PAIRS:
        for method in methods:
            pair_key = f"{model_a}_vs_{model_b}/{method}"
            if not args.force and any(
                r["pair"] == f"{model_a}_vs_{model_b}" and r["method"] == method
                for r in existing_cross
            ):
                _log(f"[skip] cross {pair_key}")
                continue
            _log(f"[run] cross {pair_key}")
            new_cross.extend(compute_cross_model(model_a, model_b, method))
            _log(f"[done] cross {pair_key}")

    # Merge and save
    all_within = existing_within + new_within
    all_cross = existing_cross + new_cross
    io.save_jsonl(all_within, within_out)
    io.save_jsonl(all_cross, cross_out)
    print(f"[ok] Saved:\n  {within_out}\n  {cross_out}")

    # Summary JSON (flat dict for convenience, keyed by "model/method/n=N")
    full_summary = {}
    for r in all_within:
        full_summary[build_summary_key(r)] = {k: r[k] for k in
            ["median", "q1", "q3", "whisker_min", "whisker_max", "mean", "count"]}
    for r in all_cross:
        full_summary[build_summary_key(r)] = {k: r[k] for k in
            ["median", "q1", "q3", "whisker_min", "whisker_max", "mean", "count"]}
    io.save_json(full_summary, summary_out)
    print(f"[ok] Saved {summary_out}")

    # Figures
    make_plots(all_within, all_cross)

    # Print quick summary to stdout
    def _fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    print("\n=== Within-Model Cost-Effectiveness Summary ===")
    for r in all_within:
        print(f"  {r['model']}/{r['method']}/n={r['n']}: "
              f"median={_fmt(r['median'])}  mean={_fmt(r['mean'])}  "
              f"n_bugs={r['count']}")

    print("\n=== Cross-Model Cost-Effectiveness Summary ===")
    for r in all_cross:
        print(f"  {r['pair']}/{r['method']}/n={r['n']}: "
              f"median={_fmt(r['median'])}  mean={_fmt(r['mean'])}  "
              f"n_bugs={r['count']}")


if __name__ == "__main__":
    main()
