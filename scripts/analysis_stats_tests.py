#!/usr/bin/env python3
"""
Statistical significance tests for convergence-correctness analysis.

Performs Fisher's exact test for key contrasts across all models:
1. Within-model: per-model syntactic and semantic 1/6 vs 6/6
2. Cross-model: per-pair cross-model semantic 1/6 vs 6/6
3. Difficulty: per-model difficulty (Easy/Medium/Hard) 1/6 vs 6/6

Usage:
    cd scripts/
    python3 analysis_stats_tests.py
"""

import json
import csv
from scipy.stats import fisher_exact

import analysis_config as cfg

DATA_DIR = str(cfg.ANALYSIS_DIR)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fisher_test_convergence(records, model, method, bucket_a, bucket_b, value_field="resolved_all"):
    """Compare two convergence buckets using Fisher's exact test.

    Contingency table:
                    resolved    not resolved
    bucket_a        a           b
    bucket_b        c           d
    """
    group_a = [r for r in records if r["model"] == model and r["method"] == method and r["bucket"] == bucket_a]
    group_b = [r for r in records if r["model"] == model and r["method"] == method and r["bucket"] == bucket_b]

    a = sum(1 for r in group_a if r[value_field])
    b = len(group_a) - a
    c = sum(1 for r in group_b if r[value_field])
    d = len(group_b) - c

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    return {
        "model": model,
        "method": method,
        "bucket_a": bucket_a,
        "bucket_b": bucket_b,
        "n_a": len(group_a),
        "resolved_a": a,
        "n_b": len(group_b),
        "resolved_b": c,
        "precision_a": a / len(group_a) if group_a else 0,
        "precision_b": c / len(group_b) if group_b else 0,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def fisher_test_cross_model(pair, bucket_a, bucket_b):
    """Compare two cross-model agreement buckets for a given pair."""
    model_a, model_b = pair
    pair_key = f"{model_a}_vs_{model_b}"
    path = f"{DATA_DIR}/cross_model_semantic.jsonl"
    records = load_jsonl(path)

    pair_records = [r for r in records if r["pair"] == pair_key]

    group_a = [r for r in pair_records if r["bucket"] == bucket_a]
    group_b = [r for r in pair_records if r["bucket"] == bucket_b]

    if not group_a and not group_b:
        return None

    a = sum(1 for r in group_a if r["resolved"])
    b = len(group_a) - a
    c = sum(1 for r in group_b if r["resolved"])
    d = len(group_b) - c

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    return {
        "model": f"cross-model ({model_a} vs {model_b})",
        "method": "semantic",
        "bucket_a": bucket_a,
        "bucket_b": bucket_b,
        "n_a": len(group_a),
        "resolved_a": a,
        "n_b": len(group_b),
        "resolved_b": c,
        "precision_a": a / len(group_a) if group_a else 0,
        "precision_b": c / len(group_b) if group_b else 0,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def load_difficulty_data():
    """Load difficulty cross-tab data for all models."""
    results = {}
    for model in cfg.WITHIN_MODEL_MODELS:
        path = f"{DATA_DIR}/difficulty_crosstab_{model.replace('-', '_').replace(' ', '_')}.csv"
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (model, row["difficulty_short"], row["bucket"])
                    results[key] = {
                        "total": int(row["total"]),
                        "resolved_all": int(row["resolved_all"]),
                    }
        except FileNotFoundError:
            pass
    return results


def fisher_test_difficulty(model, difficulty, bucket_a, bucket_b):
    """Compare two convergence buckets within a difficulty tier for a given model."""
    path = f"{DATA_DIR}/difficulty_crosstab_{model.replace('-', '_').replace(' ', '_')}.csv"

    def get_row(diff, bucket):
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["difficulty_short"] == diff and row["bucket"] == bucket:
                        return int(row["total"]), int(row["resolved_all"])
        except FileNotFoundError:
            pass
        return 0, 0

    total_a, resolved_a = get_row(difficulty, bucket_a)
    total_b, resolved_b = get_row(difficulty, bucket_b)

    a = resolved_a
    b = total_a - resolved_a
    c = resolved_b
    d = total_b - resolved_b

    if total_a == 0 or total_b == 0:
        return None

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    return {
        "model": model,
        "difficulty": difficulty,
        "bucket_a": bucket_a,
        "bucket_b": bucket_b,
        "n_a": total_a,
        "resolved_a": a,
        "n_b": total_b,
        "resolved_b": c,
        "precision_a": a / total_a,
        "precision_b": c / total_b,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS FOR CONVERGENCE-CORRECTNESS ANALYSIS")
    print("=" * 80)

    # Load convergence data
    conv_records = load_jsonl(f"{DATA_DIR}/convergence_syntactic.jsonl")
    beh_records = load_jsonl(f"{DATA_DIR}/convergence_semantic.jsonl")

    tests = []

    # --- Within-model convergence ---
    print("\n--- Within-Model Convergence ---\n")

    for model in cfg.WITHIN_MODEL_MODELS:
        for method, records in [("syntactic", conv_records), ("semantic", beh_records)]:
            result = fisher_test_convergence(records, model, method, "1/6", "6/6")
            if result["n_a"] == 0 and result["n_b"] == 0:
                continue
            test_name = f"within-model {model} {method} 1/6 vs 6/6"
            tests.append((test_name, result))
            print(f"{model} | {method} | 1/6 vs 6/6")
            print(f"  1/6: {result['resolved_a']}/{result['n_a']} = {result['precision_a']:.1%}")
            print(f"  6/6: {result['resolved_b']}/{result['n_b']} = {result['precision_b']:.1%}")
            print(f"  Fisher's exact test: OR={result['odds_ratio']:.2f}, p={result['p_value']:.2e}")
            print()

    # --- Cross-model agreement ---
    print("\n--- Cross-Model Agreement ---\n")

    for pair in cfg.CROSS_MODEL_PAIRS:
        result = fisher_test_cross_model(pair, "1/6", "6/6")
        if result is None:
            print(f"cross-model ({pair[0]} vs {pair[1]}) | semantic | 1/6 vs 6/6: no data")
            continue
        test_name = f"cross-model {pair[0]} vs {pair[1]} semantic 1/6 vs 6/6"
        tests.append((test_name, result))
        print(f"cross-model ({pair[0]} vs {pair[1]}) | semantic | 1/6 vs 6/6")
        print(f"  1/6: {result['resolved_a']}/{result['n_a']} = {result['precision_a']:.1%}")
        print(f"  6/6: {result['resolved_b']}/{result['n_b']} = {result['precision_b']:.1%}")
        print(f"  Fisher's exact test: OR={result['odds_ratio']:.2f}, p={result['p_value']:.2e}")
        print()

    # --- Convergence vs difficulty ---
    print("\n--- Convergence vs Difficulty ---\n")

    for model in cfg.WITHIN_MODEL_MODELS:
        for diff in ["Easy", "Medium", "Hard"]:
            result = fisher_test_difficulty(model, diff, "1/6", "6/6")
            if result is None:
                print(f"{model} | {diff} | 1/6 vs 6/6: insufficient data")
                continue
            test_name = f"difficulty {model} {diff} 1/6 vs 6/6"
            tests.append((test_name, result))
            print(f"{model} | {diff} | 1/6 vs 6/6")
            print(f"  1/6: {result['resolved_a']}/{result['n_a']} = {result['precision_a']:.1%}")
            print(f"  6/6: {result['resolved_b']}/{result['n_b']} = {result['precision_b']:.1%}")
            print(f"  Fisher's exact test: OR={result['odds_ratio']:.2f}, p={result['p_value']:.2e}")
            print()

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Test':<45} {'p-value':<12} {'Sig.':<8} {'OR':<8}")
    print("-" * 80)
    for name, r in tests:
        if r is None:
            continue
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{name:<45} {r['p_value']:<12.2e} {sig:<8} {r['odds_ratio']:<8.2f}")
    print("-" * 80)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print()

    # Save results to JSON for paper reference
    output_path = f"{DATA_DIR}/stats_test_results.json"
    serializable = {}
    for name, r in tests:
        if r is not None:
            serializable[name] = {k: v for k, v in r.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
