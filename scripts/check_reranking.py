#!/usr/bin/env python3
import json

def check_variant(name, no_rerank_file, rerank_file):
    print(f"\n{'='*70}")
    print(f"Checking {name.upper()}")
    print(f"{'='*70}")
    
    with open(no_rerank_file) as f:
        data_no_rerank = [json.loads(line) for line in f]
    
    with open(rerank_file) as f:
        data_rerank = [json.loads(line) for line in f]
    
    same_count = 0
    changed_count = 0
    
    for i in range(min(5, len(data_no_rerank))):
        indices_before = data_no_rerank[i]["retrieved_indices"][:5]
        indices_after = data_rerank[i]["retrieved_indices"][:5]
        is_same = indices_before == indices_after
        
        if is_same:
            same_count += 1
        else:
            changed_count += 1
        
        print(f"\nQ{i+1}: {data_no_rerank[i]['question'][:55]}...")
        print(f"  Before: {indices_before}")
        print(f"  After:  {indices_after}")
        print(f"  Changed: {'✅ YES' if not is_same else '❌ NO (SAME)'}")
    
    print(f"\nSummary: {changed_count} changed, {same_count} unchanged (out of first 5)")
    return changed_count > 0

# Check each variant
print("\n🔍 CHECKING IF RERANKING IS ACTUALLY CHANGING RESULTS")

baseline_changed = check_variant(
    "baseline",
    "results/rag_compare/baseline/baseline_details.jsonl",
    "results/rag_compare_rerank_cross_encoder/baseline+rerank_details.jsonl"
)

finance_changed = check_variant(
    "finance",
    "results/rag_compare/finance/finance_details.jsonl",
    "results/rag_compare_rerank_cross_encoder/finance+rerank_details.jsonl"
)

hybrid_changed = check_variant(
    "hybrid",
    "results/rag_compare/hybrid/hybrid_details.jsonl",
    "results/rag_compare_rerank_cross_encoder/hybrid+rerank_details.jsonl"
)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Baseline: {'✅ Reranking IS working' if baseline_changed else '❌ Reranking NOT working'}")
print(f"Finance:  {'✅ Reranking IS working' if finance_changed else '❌ Reranking NOT working'}")
print(f"Hybrid:   {'✅ Reranking IS working' if hybrid_changed else '❌ Reranking NOT working'}")
