#!/usr/bin/env python3

"""
python3 analysis_complementarity.py

Expected layout:
  <run_root>/run_<it>/<model>.<rest>.json
  Model name = first dot-segment of the JSON filename.
  All run_<N> directories under <run_root> are auto-discovered (any N).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import analysis_config as cfg
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


TOTAL = 500


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_reports(run_root: Path) -> List[Tuple[str, int, Path]]:
    found: List[Tuple[str, int, Path]] = []
    for run_dir in sorted(run_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        suffix = run_dir.name[len("run_"):]
        if not suffix.isdigit():
            continue
        it = int(suffix)
        for report in sorted(run_dir.glob("*.json")):
            if not report.is_file():
                continue
            model = report.name.split(".", 1)[0]
            found.append((model, it, report))
    found.sort(key=lambda x: (x[0], x[1]))
    return found

def pct(n: int, denom: int = TOTAL) -> float:
    return (n / denom) if denom else 0.0


def pct_str(n: int, denom: int = TOTAL) -> str:
    return f"{pct(n, denom) * 100:.1f}%"


def write_summary_csv(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "iteration",
        "resolved", "unresolved", "resolved_rate",
        "new_solved", "repeat_solved", "new_rate", "repeat_rate",
        "report_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def annotate_bar_segments(ax, x: float, bottom: float, height: float, label: str) -> None:
    if height <= 0:
        return
    ax.text(x, bottom + height / 2.0, label, ha="center", va="center", fontsize=cfg.PLOT_FONT_BAR_INSIDE)


def main() -> None:
    
    if not cfg.RUN_ROOT.is_dir():
        print(cfg.RUN_ROOT,"does not exist. Exiting...")
        return None
    # Ensure analysis directory exists
    cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = cfg.ANALYSIS_DIR / "summary_complementarity.csv"
    out_path = cfg.ANALYSIS_DIR / "paper_complementarity_bars.png"

    reports = find_reports(cfg.RUN_ROOT)
    if not reports:
        raise SystemExit("No reports found. Check run_root and the expected naming pattern.")

    # Load solved sets per (model, iter)
    solved_sets: Dict[str, Dict[int, set]] = {}
    report_paths: Dict[Tuple[str, int], str] = {}

    for model, it, path in reports:
        d = load_json(path)

        # Optional sanity check: if file says total_instances, ensure it matches our fixed reference point.
        file_total = d.get("total_instances", TOTAL)
        if int(file_total) != TOTAL:
            print(f"[WARN] {path}: total_instances={file_total} != {TOTAL}. Using {TOTAL} as reference.")

        resolved_ids = {str(x) for x in d.get("resolved_ids", [])}
        solved_sets.setdefault(model, {})[it] = resolved_ids
        report_paths[(model, it)] = str(path)

    # Build CSV rows with complementarity (new vs repeat) based on resolved_ids sets
    rows: List[dict] = []
    for model in sorted(solved_sets.keys()):
        solved_before: set = set()  # union of all earlier iterations for this model
        for it in sorted(solved_sets[model].keys()):
            solved_now = solved_sets[model][it]

            resolved = len(solved_now)
            unresolved = TOTAL - resolved

            newly_fixed_count = len(solved_now - solved_before)
            fixed_before_count = resolved - newly_fixed_count  # derived: total - newly

            rows.append({
                "model": model,
                "iteration": it,
                "resolved": resolved,
                "unresolved": unresolved,
                "resolved_rate": round(pct(resolved), 6),

                "new_solved": newly_fixed_count,
                "repeat_solved": fixed_before_count,
                "new_rate": round(pct(newly_fixed_count), 6),
                "repeat_rate": round(pct(fixed_before_count), 6),

                "report_path": report_paths.get((model, it), ""),
            })

            solved_before |= solved_now

    rows.sort(key=lambda r: (r["model"], r["iteration"]))

    # 1) CSV output (no terminal table distortion)
    write_summary_csv(rows, out_csv)
    print(f"[OK] Wrote CSV: {out_csv.resolve()}")

    # 2) Plots

    # Single consolidated complementarity plot:
    #   x-axis groups = models. Within each model's group, the bars are
    #   "Run 1", "Run 2", ..., "Run N", "Total".
    #   Per-run bars are stacked: darker shade = repeat (fixed in earlier run(s)),
    #   lighter shade = newly fixed in this run. Color = iteration.
    #   The "Total" bar per model is the cumulative union of resolved_ids across
    #   all that model's runs (single solid bar, no internal split — every bug
    #   in the union is counted once).
    

    # Custom display order for models on the x-axis. Known models go first in this
    # order; any unknown model falls back to alphabetical at the end.
    PREFERRED_MODEL_ORDER = cfg.WITHIN_MODEL_MODELS

    def _model_sort_key(m: str):
        try:
            return (0, PREFERRED_MODEL_ORDER.index(m))
        except ValueError:
            return (1, m)

    models = sorted(solved_sets.keys(), key=_model_sort_key)
    all_iters = sorted({it for m in solved_sets for it in solved_sets[m].keys()})
    rows_by = {(r["model"], r["iteration"]): r for r in rows}

    # Cumulative union of resolved_ids per model (across that model's runs).
    totals_per_model = {
        m: len(set().union(*solved_sets[m].values())) if solved_sets[m] else 0
        for m in models
    }

    cmap = plt.get_cmap("tab10")
    iter_color = {it: cmap(i % 10) for i, it in enumerate(all_iters)}
    total_color = (0.35, 0.35, 0.35, 1.0)  # dark gray for the Total bar

    def _shade(rgba, factor: float):
        r, g, b, a = rgba
        if factor < 1.0:
            return (r * factor, g * factor, b * factor, a)
        return (
            min(1.0, r + (1 - r) * (factor - 1)),
            min(1.0, g + (1 - g) * (factor - 1)),
            min(1.0, b + (1 - b) * (factor - 1)),
            a,
        )

    # Lay out only the runs each model actually has, plus a Total bar, then a gap.
    gap_between_groups = 1                 # empty slot
    bar_w = 0.8                            # actual bar width (slots are 1.0 wide)

    xtick_positions: List[float] = []
    xtick_labels: List[str] = []
    positions_by_model: Dict[str, Tuple[List[float], List[int]]] = {}
    group_centers: List[float] = []

    cursor = 0.0
    for model in models:
        model_iters = sorted(solved_sets[model].keys())
        pos: List[float] = []
        for i, it in enumerate(model_iters):
            pos.append(cursor)
            xtick_labels.append(f"Run {it}")
            xtick_positions.append(cursor)
            cursor += 1.0

        # Cumulative Total slot at the end of the model's group
        pos.append(cursor)
        xtick_positions.append(cursor)
        xtick_labels.append("Total")
        cursor += 1.0

        positions_by_model[model] = (pos, model_iters)
        group_centers.append((pos[0] + pos[-1]) / 2.0)
        cursor += gap_between_groups       # gap before next model group

    # Split models into two rows for readability
    split = (len(models) + 1) // 2
    model_rows = [models[:split], models[split:]]

    fig, axes = plt.subplots(2, 1, figsize=(40, 10), sharey=True)
    fig.subplots_adjust(hspace=0.45)

    for row_idx, (ax, row_models) in enumerate(zip(axes, model_rows)):
        # Recompute positions local to this row
        row_xtick_pos = []
        row_xtick_lbl = []
        row_group_centers = []
        row_cursor = 0.0
        row_positions: Dict[str, Tuple[List[float], List[int]]] = {}

        for model in row_models:
            model_iters = sorted(solved_sets[model].keys())
            pos: List[float] = []
            for i, it in enumerate(model_iters):
                pos.append(row_cursor)
                row_xtick_lbl.append(f"Run {it}")
                row_xtick_pos.append(row_cursor)
                row_cursor += 1.0
            pos.append(row_cursor)
            row_xtick_pos.append(row_cursor)
            row_xtick_lbl.append("Total")
            row_cursor += 1.0
            row_positions[model] = (pos, model_iters)
            row_group_centers.append((pos[0] + pos[-1]) / 2.0)
            row_cursor += gap_between_groups

        for model in row_models:
            pos, model_iters = row_positions[model]
            for ti, it in enumerate(model_iters):
                r = rows_by.get((model, it))
                if not r:
                    continue
                x = pos[ti]
                base_color = iter_color[it]
                rep = r["repeat_solved"]
                new = r["new_solved"]
                total = rep + new

                ax.bar(x, rep, bar_w, color=_shade(base_color, 0.6))
                ax.bar(x, new, bar_w, bottom=rep, color=_shade(base_color, 1.5))

                # if rep > 0:
                #     ax.text(x, rep / 2.0, f"{rep}\n{pct_str(rep)}",
                #             ha="center", va="center", fontsize=cfg.PLOT_FONT_BAR_INSIDE, color="white")
                # if new > 0:
                #     ax.text(x, rep + new / 2.0, f"{new}\n{pct_str(new)}",
                #             ha="center", va="top", fontsize=cfg.PLOT_FONT_BAR_INSIDE, color="white")
                if rep > 0:
                    ax.text(x, rep / 2.0, f"{rep}",
                            ha="center", va="center", fontsize=cfg.PLOT_FONT_BAR_INSIDE, color="white")
                if new > 0:
                    ax.text(x, rep + new / 2.0, f"{new}",
                            ha="center", va="top", fontsize=cfg.PLOT_FONT_BAR_INSIDE, color="white")
                if total > 0:
                    ax.text(x, total, f"{total}\n{pct_str(total)}",
                            ha="center", va="bottom", fontsize=cfg.PLOT_FONT_BAR_ABOVE)

            t = totals_per_model.get(model, 0)
            if t > 0:
                x = pos[-1]
                ax.bar(x, t, bar_w, color=total_color)
                ax.text(x, t, f"{t}\n{pct_str(t)}", ha="center", va="bottom", fontsize=cfg.PLOT_FONT_BAR_ABOVE)

        ax.set_xticks(row_xtick_pos)
        ax.set_xticklabels(row_xtick_lbl, rotation=0, fontsize=cfg.PLOT_FONT_TICK)
        ax.tick_params(axis="y", labelsize=cfg.PLOT_FONT_TICK)
        ax.set_ylabel("Resolved bugs", fontsize=cfg.PLOT_FONT_AXIS_LABEL)
        ax.set_ylim(0, TOTAL)
        ax.grid(True, axis="y", linewidth=0.3)
        ax.set_xlim(-0.5, row_cursor - gap_between_groups + 0.5)

        for center, model in zip(row_group_centers, row_models):
            ax.annotate(model, xy=(center, -0.13), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=cfg.PLOT_FONT_TICK)

    iter_handles = [mpatches.Patch(color=iter_color[it], label=f"Run {it}") for it in all_iters]
    total_handle = [mpatches.Patch(color=total_color, label="Total (cumulative union)")]
    shade_handles = [
        mpatches.Patch(facecolor=(0.35, 0.35, 0.35), label="darker = fixed in earlier run(s)"),
        mpatches.Patch(facecolor=(0.85, 0.85, 0.85), label="lighter = newly fixed (incremental)"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(hspace=0.65)
    fig.legend(handles=iter_handles + total_handle + shade_handles,
               loc="center", ncol=5, fontsize=cfg.PLOT_FONT_LEGEND, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.50))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Plot: {out_path.resolve()}")

    print(f"\nAll outputs saved under: {cfg.ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
