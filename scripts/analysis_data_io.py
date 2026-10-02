#!/usr/bin/env python3
"""
Data I/O: load patches, reports, and summary JSONs one run at a time.
Save intermediate results as human-readable JSONL/JSON files.
Check for existing intermediates before reprocessing.
"""

import json
from pathlib import Path

import analysis_config as cfg


# ---------------------------------------------------------------------------
# Loading raw data
# ---------------------------------------------------------------------------

def load_summary(run_id: int, model: str) -> dict | None:
    """
    Load the summary JSON for a given run and model.
    Looks for <RUN_ROOT>/run_<run_id>/<model>.preds_*.json
    Returns None if not found.
    """
    run_dir = cfg.RUN_ROOT / f"run_{run_id}"
    if not run_dir.is_dir():
        return None
    for f in run_dir.glob(f"{model}.preds_*.json"):
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def load_patch(run_id: int, model: str, instance_id: str) -> str | None:
    """
    Load patch.diff text for a given run, model, and instance.
    Returns None if not found.
    """
    preds_dirs = list(
        (cfg.RUN_ROOT / f"run_{run_id}" / "logs" / "run_evaluation").glob(
            f"preds_SWE-bench_Verified_{model}_run_*"
        )
    )
    for preds_dir in preds_dirs:
        patch_file = preds_dir / model / instance_id / "patch.diff"
        if patch_file.is_file():
            return patch_file.read_text(encoding="utf-8")
    return None


def load_report(run_id: int, model: str, instance_id: str) -> dict | None:
    """
    Load report.json for a given run, model, and instance.
    Returns None if not found.
    """
    preds_dirs = list(
        (cfg.RUN_ROOT / f"run_{run_id}" / "logs" / "run_evaluation").glob(
            f"preds_SWE-bench_Verified_{model}_run_*"
        )
    )
    for preds_dir in preds_dirs:
        report_file = preds_dir / model / instance_id / "report.json"
        if report_file.is_file():
            with open(report_file, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def load_all_instance_ids(run_id: int, model: str) -> list[str]:
    """
    Get all instance_ids that have a patch.diff or report.json for a given run and model.
    """
    preds_dirs = list(
        (cfg.RUN_ROOT / f"run_{run_id}" / "logs" / "run_evaluation").glob(
            f"preds_SWE-bench_Verified_{model}_run_*"
        )
    )
    instance_ids = []
    for preds_dir in preds_dirs:
        model_dir = preds_dir / model
        if model_dir.is_dir():
            for inst_dir in sorted(model_dir.iterdir()):
                if (inst_dir / "report.json").is_file() or (inst_dir / "patch.diff").is_file():
                    instance_ids.append(inst_dir.name)
    return instance_ids


# ---------------------------------------------------------------------------
# Saving / loading intermediates
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Returns empty list if missing."""
    if not path.is_file():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: dict, path: Path) -> None:
    """Write a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict | None:
    """Read a JSON file. Returns None if missing."""
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def intermediate_exists(path: Path) -> bool:
    """Check if an intermediate file already exists and is non-empty."""
    return path.is_file() and path.stat().st_size > 0
