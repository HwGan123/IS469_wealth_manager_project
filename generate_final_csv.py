#!/usr/bin/env python3
import json
import csv
from pathlib import Path

results_dir = Path("results")
baseline_dir = results_dir / "rag_compare"
rerank_dir = results_dir / "rag_compare_rerank_cross_encoder"

variant_order = ["baseline", "hyde", "hybrid", "finance"]

all_results = []

# Load baseline (no rerank)
for variant_dir in baseline_dir.glob("*/"):
    if variant_dir.is_dir():
        summary_file = variant_dir / "comparison_summary.json"
        if summary_file.exists():
            with summary_file.open("r") as f:
                data = json.load(f)
                for row in data:
                    row.pop("faithfulness", None)
                    row.pop("context_precision", None)
                    row.pop("context_recall", None)
                    row.pop("hallucination_rate", None)
                    all_results.append(row)

# Load rerank results
rerank_summary = rerank_dir / "comparison_summary.json"
if rerank_summary.exists():
    with rerank_summary.open("r") as f:
        rerank_data = json.load(f)
        for row in rerank_data:
            row.pop("faithfulness", None)
            row.pop("context_precision", None)
            row.pop("context_recall", None)
            row.pop("hallucination_rate", None)
            all_results.append(row)

# Sort by variant order
def get_sort_key(row):
    variant = row["variant"]
    base_variant = variant.replace("+rerank", "")
    try:
        order_idx = variant_order.index(base_variant)
    except ValueError:
        order_idx = len(variant_order)
    is_rerank = 1 if "+rerank" in variant else 0
    return (order_idx, is_rerank)

all_results.sort(key=get_sort_key)

# Write with header
output_path = results_dir / "all_results_comparison.csv"
fieldnames = ["variant", "num_questions", "precision_at_k", "recall_at_k", "f1_score", "accuracy", "mrr"]

with output_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        row.pop("reranking", None)
        writer.writerow(row)

print("✅ CSV generated with header")
print(f"Saved to: {output_path}\n")

# Display results
print("Results (ordered: baseline, hyde, hybrid, finance):")
print("-" * 100)
for row in all_results:
    print(f"{row['variant']:20} F1: {row['f1_score']:.4f} | Precision: {row['precision_at_k']:.3f} | Recall: {row['recall_at_k']:.3f} | MRR: {row['mrr']:.4f}")
