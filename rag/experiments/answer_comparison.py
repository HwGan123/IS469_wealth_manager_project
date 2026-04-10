import json
import csv

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Load all variants
variants = {
    "baseline":        load_jsonl("results/rag_compare_ragas/baseline_details.jsonl"),
    "hyde":            load_jsonl("results/rag_compare_ragas/hyde_details.jsonl"),
    "hybrid":          load_jsonl("results/rag_compare_ragas/hybrid_details.jsonl"),
    "finance":         load_jsonl("results/rag_compare_ragas/finance_details.jsonl"),
    "baseline+rerank": load_jsonl("results/rag_compare_ragas_rerank/baseline+rerank_details.jsonl"),
    "hyde+rerank":     load_jsonl("results/rag_compare_ragas_rerank/hyde+rerank_details.jsonl"),
    "hybrid+rerank":   load_jsonl("results/rag_compare_ragas_rerank/hybrid+rerank_details.jsonl"),
    "finance+rerank":  load_jsonl("results/rag_compare_ragas_rerank/finance+rerank_details.jsonl"),
}

# Helper to check if answered
def is_answered(answer: str) -> bool:
    if not answer:
        return False
    answer = answer.lower()
    keywords = [
        "insufficient evidence",
        "cannot answer",
        "not provided",
        "does not provide",
        "not enough information"
    ]
    return not any(k in answer for k in keywords)

# Number of questions
N = 5
reference_key = "hybrid"
reference = variants[reference_key]

# Output CSV file
output_file = "rag_answer_comparison.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Header
    writer.writerow([
        "question_id",
        "question",
        "variant",
        "answered",
        "answer"
    ])

    # Rows
    for i in range(N):
        question = reference[i]["question"]

        for name, data in variants.items():
            ans = data[i]["answer"]
            answered_flag = is_answered(ans)

            writer.writerow([
                f"Q{i+1}",
                question,
                name,
                answered_flag,
                ans
            ])

print(f"CSV file saved to: {output_file}")