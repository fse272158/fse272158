#!/usr/bin/env python3
"""
Patch similarity methods — pure functions, no side effects.

Three methods:
1. Exact match: strip whitespace/blank lines, then string equality
2. Syntactic match: same as exact + normalize variable names
3. Semantic match: compare test outcomes (FAIL_TO_PASS + PASS_TO_PASS)
"""

import re


# ---------------------------------------------------------------------------
# Preprocessing (shared by exact and syntactic)
# ---------------------------------------------------------------------------

def _preprocess_diff(diff_text: str) -> str:
    """
    Shared preprocessing for exact and syntactic methods:
    - Strip leading/trailing whitespace per line
    - Normalize multiple blank lines to a single blank line
    - Remove trailing blank lines
    """
    lines = diff_text.splitlines()
    stripped = [line.strip() for line in lines]
    # Collapse multiple blank lines into one
    result = []
    prev_blank = False
    for line in stripped:
        if line == "":
            if not prev_blank:
                result.append(line)
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    # Remove trailing blank lines
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Variable name normalization
# ---------------------------------------------------------------------------

# Pattern: single-char identifiers (common temp variable names)
_SINGLE_CHAR_VAR = re.compile(r"\b([a-zA-Z_])\b")

# Pattern: short numeric suffixes like var1, var2, tmp_0, etc.
_SHORT_SUFFIX_VAR = re.compile(r"\b([a-zA-Z]+)_?\d+\b")


def _normalize_variable_names(diff_text: str) -> str:
    """
    Normalize variable names in diff text to reduce noise from
    different naming choices across runs.
    - Single-char variables → _VAR_
    - Variables with numeric suffixes (var1, var2, tmp_0) → _VAR_
    """
    text = _SINGLE_CHAR_VAR.sub("_VAR_", diff_text)
    text = _SHORT_SUFFIX_VAR.sub("_VAR_", text)
    return text


# ---------------------------------------------------------------------------
# Public similarity methods
# ---------------------------------------------------------------------------

def exact_match(patch_a: str, patch_b: str) -> bool:
    """
    Exact match: preprocess both diffs (strip whitespace, normalize blank lines),
    then compare for string equality.
    """
    return _preprocess_diff(patch_a) == _preprocess_diff(patch_b)


def syntactic_match(patch_a: str, patch_b: str) -> bool:
    """
    Syntactic match: same as exact match, plus normalize variable names.
    Catches semantically identical patches that differ only in variable naming.
    """
    norm_a = _normalize_variable_names(_preprocess_diff(patch_a))
    norm_b = _normalize_variable_names(_preprocess_diff(patch_b))
    return norm_a == norm_b


def semantic_match(report_a: dict, report_b: dict) -> bool:
    """
    Semantic match: two patches are "same" if they produce identical
    test outcomes. Compares both FAIL_TO_PASS and PASS_TO_PASS success/failure
    sets from report.json.

    A patch that passes FAIL_TO_PASS but breaks PASS_TO_PASS is a regression
    and should NOT be considered equivalent to one that passes both.
    """
    # Extract test status sets from reports
    # report format: { "<instance_id>": { "tests_status": { ... } } }
    status_a = _extract_tests_status(report_a)
    status_b = _extract_tests_status(report_b)

    if status_a is None or status_b is None:
        return False

    return status_a == status_b


def _extract_tests_status(report: dict) -> dict | None:
    """
    Extract FAIL_TO_PASS and PASS_TO_PASS success/failure sets from a report.
    Returns a comparable dict, or None if the report format is unexpected.
    """
    if not isinstance(report, dict):
        return None

    # report.json format: { "<instance_id>": { "tests_status": {...} } }
    # Get the first (only) top-level value
    instance_data = next(iter(report.values()), None)
    if not isinstance(instance_data, dict):
        return None

    tests_status = instance_data.get("tests_status")
    if not isinstance(tests_status, dict):
        return None

    result = {}
    for key in ["FAIL_TO_PASS", "PASS_TO_PASS"]:
        section = tests_status.get(key, {})
        success = frozenset(section.get("success", []))
        failure = frozenset(section.get("failure", []))
        result[key] = {"success": success, "failure": failure}

    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compare_patches(
    patch_a: str | None,
    patch_b: str | None,
    report_a: dict | None,
    report_b: dict | None,
    method: str,
) -> bool:
    """
    Compare two patches using the specified similarity method.
    Returns True if the patches are considered "the same".

    Args:
        patch_a, patch_b: patch.diff text (may be None if missing)
        report_a, report_b: report.json data (may be None if missing)
        method: one of "exact", "syntactic", "semantic"
    """
    if method == "exact":
        if patch_a is None or patch_b is None:
            return False
        return exact_match(patch_a, patch_b)

    elif method == "syntactic":
        if patch_a is None or patch_b is None:
            return False
        return syntactic_match(patch_a, patch_b)

    elif method == "semantic":
        if report_a is None or report_b is None:
            return False
        return semantic_match(report_a, report_b)

    else:
        raise ValueError(f"Unknown similarity method: {method}")
