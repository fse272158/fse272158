#!/usr/bin/env python3
"""
Publication-quality bar charts for convergence-correctness analysis.

Two figures, same style:

1. Within-model convergence (paper_convergence_bars.png):
   - Stacked bars: resolved (bottom) + unresolved (top) at each convergence level
   - X-axis: convergence buckets 1/6 to 6/6
   - Two models side by side (blue = gpt-5-mini, purple = claude-opus-4-7)
   - Panels: syntactic vs semantic

2. Cross-model agreement (paper_cross_model_bars.png):
   - Same stacked bar style
   - X-axis: agreement buckets 1/6 to 6/6
   - Single model pair: gpt-5-mini vs claude-opus-4-7
   - Panels: syntactic vs semantic

Bucket meaning (both figures):
- 1/6 = only 1 run produced the same patch (no convergence / minimal agreement)
- 6/6 = all 6 runs produced the same patch (full convergence / full agreement)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

import analysis_config as cfg
import analysis_data_io as io


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL = 500  # fixed reference point for SWE-bench Verified

PLOT_METHODS = ["syntactic", "semantic"]

METHOD_LABELS = {
    "syntactic": "Syntactic comparison",
    "semantic": "Semantic comparison",
}

MODEL_LABELS = {
    "gpt-5-mini": "GPT-5-mini",
    "claude-opus-4-7": "Claude-Opus-4-7",
}

# Color scheme: tab10 colormap, same approach as script_to_present_results.py
# Each model gets a distinct hue; resolved = darker shade, unresolved = lighter shade
_cmap = plt.get_cmap("tab10")
_MODEL_TAB10_IDX = {
    "gpt-5-mini": 0,
    "claude-opus-4-7": 1,
}


def _shade(rgba, factor: float):
    """Lighten (>1.0) or darken (<1.0) an RGBA color."""
    r, g, b, a = rgba
    if factor < 1.0:
        return (r * factor, g * factor, b * factor, a)
    return (
        min(1.0, r + (1 - r) * (factor - 1)),
        min(1.0, g + (1 - g) * (factor - 1)),
        min(1.0, b + (1 - b) * (factor - 1)),
        a,
    )


def _model_colors(model: str):
    """Return (resolved_color, unresolved_color, edge_color) for a model using tab10."""
    base = _cmap(_MODEL_TAB10_IDX.get(model, 0))
    return _shade(base, 0.6), _shade(base, 1.5), _shade(base, 0.4)


# Cross-model uses a single tab10 hue (green, index=2) for the combined pair
_CROSS_BASE = _cmap(4)
CROSS_MODEL_COLORS = {
    "resolved": _shade(_CROSS_BASE, 0.6),
    "unresolved": _shade(_CROSS_BASE, 1.5),
    "edge": _shade(_CROSS_BASE, 0.4),
}


def pct_str(n: int, denom: int = TOTAL) -> str:
    """Return a percentage string like '24.0%'."""
    if not denom:
        return "0.0%"
    return f"{(n / denom) * 100:.1f}%"

# Buckets in display order (increasing convergence / agreement)
BUCKET_ORDER = ["1/6", "2/6", "3/6", "4/6", "5/6", "6/6"]

BUCKET_LABELS = {
    "1/6": "1/6\n(No convergence)",
    "2/6": "2/6",
    "3/6": "3/6",
    "4/6": "4/6",
    "5/6": "5/6",
    "6/6": "6/6\n(Full convergence)",
}

BUCKET_LABELS_CROSS = {
    "1/6": "1/6\n(No agreement)",
    "2/6": "2/6",
    "3/6": "3/6",
    "4/6": "4/6",
    "5/6": "5/6",
    "6/6": "6/6\n(Full agreement)",
}



# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def _setup_style():
    """Configure matplotlib for paper-quality output."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def _plot_stacked_bars(ax, bucket_order, bucket_labels, method_data, models,
                       ylabel, title, total_line=True):
    """
    Plot stacked bars for convergence/resolution data.

    Args:
        ax: matplotlib axis
        bucket_order: list of bucket labels in display order
        bucket_labels: dict mapping bucket -> display label
        method_data: stats data for this method
        models: list of model names
        ylabel: y-axis label
        title: plot title
        total_line: whether to show horizontal line at 500
    """
    x = np.array([0, 0.8, 1.6, 2.4, 3.2, 4.0])
    width = 0.35

    for model_idx, model in enumerate(models):
        model_data = method_data.get(model, {})
        if not model_data:
            continue

        offset = (model_idx - 0.5) * width

        resolved_vals = []
        unresolved_vals = []
        total_vals = []

        for bucket in bucket_order:
            if bucket in model_data and bucket != "overall":
                d = model_data[bucket]
                total = d["total"]
                resolved = d["resolved_all"]
                unresolved = total - resolved
            else:
                total = 0
                resolved = 0
                unresolved = 0

            resolved_vals.append(resolved)
            unresolved_vals.append(unresolved)
            total_vals.append(total)

        res_color, unres_color, edge_color = _model_colors(model)

        # Unresolved (top)
        bars_unres = ax.bar(
            x + offset, unresolved_vals, width,
            bottom=resolved_vals,
            color=unres_color,
            edgecolor=edge_color,
            linewidth=0.8,
        )

        # Resolved (bottom)
        bars_res = ax.bar(
            x + offset, resolved_vals, width,
            color=res_color,
            edgecolor=edge_color,
            linewidth=0.8,
        )

        # Annotate resolved on bottom portion (count + percentage)
        for bar, res, tot in zip(bars_res, resolved_vals, total_vals):
            if res > 0:
                height = bar.get_height()
                ax.annotate(f"{res}",
                            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha="center", va="center", fontsize=14, color="white")

        # Annotate unresolved on top portion (count + percentage)
        for bar, unres, res, tot in zip(bars_unres, unresolved_vals, resolved_vals, total_vals):
            if unres > 0 and tot > 0:
                height = bar.get_height()
                ax.annotate(f"{unres}",
                            xy=(bar.get_x() + bar.get_width() / 2, res + height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha="center", va="center", fontsize=14)

        # Annotate total count + percentage above each bar
        for bar, tot, res, unres in zip(bars_res, total_vals, resolved_vals, unresolved_vals):
            top = res + unres  # total bar height = resolved + unresolved
            ax.text(bar.get_x() + bar.get_width() / 2, top,
                    f"{tot}\n{pct_str(tot)}",
                    ha="center", va="bottom", fontsize=14)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([bucket_labels[b] for b in bucket_order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 520)

    if total_line:
        ax.axhline(y=500, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Legend
    legend_handles = []
    for model in models:
        res_color, unres_color, edge_color = _model_colors(model)
        legend_handles.append(mpatches.Patch(facecolor=res_color, edgecolor=edge_color,
                                             label=f"{MODEL_LABELS.get(model, model)} (resolved)"))
        legend_handles.append(mpatches.Patch(facecolor=unres_color, edgecolor=edge_color,
                                             label=f"{MODEL_LABELS.get(model, model)} (unresolved)"))
    if total_line:
        legend_handles.append(mlines.Line2D([0], [0], color="gray", linestyle=":", linewidth=1,
                                            label=f"Total ({TOTAL})"))
    ax.legend(handles=legend_handles, loc="upper center", fontsize=16)


# ---------------------------------------------------------------------------
# Plot 1: Within-model convergence (PRIMARY PAPER FIGURE)
# ---------------------------------------------------------------------------

def plot_convergence_resolution_bars() -> None:
    """
    Stacked bar chart: convergence vs resolution for within-model.

    Two panels (syntactic, semantic). Each panel shows two models
    side by side at each convergence level.
    """
    _setup_style()

    # Load convergence stats
    convergence = {}
    for m in cfg.SIMILARITY_METHODS:
        stats_path = cfg.STATS_JSON[m]
        if stats_path.is_file():
            convergence[m] = io.load_json(stats_path)

    if not convergence:
        print("[warn] No convergence stats found")
        return

    n_methods = len(PLOT_METHODS)
    fig, axes = plt.subplots(1, n_methods, figsize=(10 * n_methods, 6), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax_idx, method in enumerate(PLOT_METHODS):
        ax = axes[ax_idx]
        method_data = convergence.get(method, {})
        if not method_data:
            continue

        _plot_stacked_bars(
            ax=ax,
            bucket_order=BUCKET_ORDER,
            bucket_labels=BUCKET_LABELS,
            method_data=method_data,
            models=cfg.WITHIN_MODEL_MODELS,
            ylabel=f"Number of Bugs (out of {TOTAL})" if ax_idx == 0 else "",
            title=METHOD_LABELS[method],
        )

    plt.tight_layout()
    path = cfg.PAPER_PLOT_CONVERGENCE_BARS
    plt.savefig(path)
    plt.close()
    print(f"[ok] Saved convergence-resolution bars -> {path}")


# ---------------------------------------------------------------------------
# Plot 2: Cross-model agreement (SECONDARY PAPER FIGURE)
# ---------------------------------------------------------------------------

def plot_cross_model_bars() -> None:
    """
    Stacked bar chart: agreement vs resolution for cross-model.

    Two panels (syntactic, semantic). Each panel shows the model pair
    gpt-5-mini vs claude-opus-4-7 at each agreement level.

    Agreement bucket definition:
    - For each run in gpt-5-mini, check if ANY patch from claude-opus-4-7 matches
    - n_matching = number of gpt-5-mini runs with at least one match
    - bucket = n_matching / 6
    """
    _setup_style()

    # Load cross-model stats
    cross_model = {}
    cross_path = cfg.CROSS_MODEL_STATS_JSON
    if cross_path.is_file():
        cross_model = io.load_json(cross_path)

    if not cross_model:
        print("[warn] No cross-model stats found")
        return

    n_methods = len(PLOT_METHODS)
    fig, axes = plt.subplots(1, n_methods, figsize=(10 * n_methods, 6), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax_idx, method in enumerate(PLOT_METHODS):
        ax = axes[ax_idx]
        method_data = cross_model.get(method, {})
        if not method_data:
            continue

        x = np.array([0, 0.6, 1.2, 1.8, 2.4, 3.0])
        width = 0.5  # single bar per bucket (one model pair)

        resolved_vals = []
        unresolved_vals = []
        total_vals = []

        for bucket in BUCKET_ORDER:
            # Find the data for this bucket (key may be tuple or string)
            d = None
            for key, value in method_data.items():
                if key == "method":
                    continue
                if isinstance(key, tuple):
                    _, b = key
                else:
                    _, b = key.split("|", 1)
                if b == bucket:
                    d = value
                    break

            if d:
                total = d["total"]
                resolved = d["resolved"]
                unresolved = total - resolved
            else:
                total = 0
                resolved = 0
                unresolved = 0

            resolved_vals.append(resolved)
            unresolved_vals.append(unresolved)
            total_vals.append(total)

        colors = CROSS_MODEL_COLORS

        # Unresolved (top)
        bars_unres = ax.bar(
            x, unresolved_vals, width,
            bottom=resolved_vals,
            color=colors["unresolved"],
            edgecolor=colors["edge"],
            linewidth=0.8,
        )

        # Resolved (bottom)
        bars_res = ax.bar(
            x, resolved_vals, width,
            color=colors["resolved"],
            edgecolor=colors["edge"],
            linewidth=0.8,
        )

        # Annotate resolved on bottom (count + percentage)
        for bar, res, tot in zip(bars_res, resolved_vals, total_vals):
            if res > 0:
                height = bar.get_height()
                ax.annotate(f"{res}",
                            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha="center", va="center", fontsize=16, color="white")

        # Annotate unresolved on top (count + percentage)
        for bar, unres, res, tot in zip(bars_unres, unresolved_vals, resolved_vals, total_vals):
            if unres > 0 and tot > 0:
                height = bar.get_height()
                ax.annotate(f"{unres}",
                            xy=(bar.get_x() + bar.get_width() / 2, res + height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha="center", va="center", fontsize=14)

        # Annotate total count + percentage above each bar
        for bar, tot, res, unres in zip(bars_res, total_vals, resolved_vals, unresolved_vals):
            top = res + unres  # total bar height = resolved + unresolved
            ax.text(bar.get_x() + bar.get_width() / 2, top,
                    f"{tot}\n{pct_str(tot)}",
                    ha="center", va="bottom", fontsize=14)

        ax.set_xticks(x)
        ax.set_xticklabels([BUCKET_LABELS_CROSS[b] for b in BUCKET_ORDER])
        ax.set_ylabel(f"Number of Bugs (out of {TOTAL})" if ax_idx == 0 else "")
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylim(0, 520)
        ax.axhline(y=TOTAL, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor=colors["resolved"], edgecolor=colors["edge"], label="Resolved"),
            mpatches.Patch(facecolor=colors["unresolved"], edgecolor=colors["edge"], label="Unresolved"),
            mlines.Line2D([0], [0], color="gray", linestyle=":", linewidth=1, label=f"Total ({TOTAL})"),
        ]
        ax.legend(handles=legend_handles, loc="upper center", fontsize=18)

    plt.tight_layout()
    path = cfg.PAPER_PLOT_CROSS_MODEL_BARS
    plt.savefig(path)
    plt.close()
    print(f"[ok] Saved cross-model bars -> {path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_plots() -> None:
    """Generate all plots."""
    print("=== Generating within-model convergence bars ===")
    plot_convergence_resolution_bars()

    print("=== Generating cross-model agreement bars ===")
    plot_cross_model_bars()


if __name__ == "__main__":
    run_plots()
