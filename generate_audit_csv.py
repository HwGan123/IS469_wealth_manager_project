#!/usr/bin/env python3
"""
Generate CSV summary from audit report JSON.
Exports variant-level statistics to a CSV table.
"""

import json
import csv
from pathlib import Path

def generate_audit_csv():
    """Generate CSV from audit report JSON."""
    
    # Read JSON report
    json_path = Path("results/auditor_compare.py/rag_audit_report.json")
    
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found")
        print("Run: .venv/bin/python audit_rag_outputs.py first")
        return
    
    with open(json_path, 'r') as f:
        report = json.load(f)
    
    # Extract variant summaries
    variants = report.get("variants_audited", [])
    
    if not variants:
        print("❌ No variants found in report")
        return
    
    # Create CSV file
    csv_path = Path("results/auditor_compare.py/audit_summary.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Variant',
            'Total Entries',
            'Overall Status',
            'Hallucination Count',
            'Hallucination Rate (%)',
            'Verified Count',
            'Unsubstantiated Count',
            'Faithfulness',
            'Answer Relevancy',
            'Context Recall',
            'Avg RAGAS Score'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write variant rows
        for variant in variants:
            ragas = variant.get("ragas_averages", {})
            faithfulness = ragas.get("faithfulness", 0)
            relevancy = ragas.get("answer_relevancy", 0)
            recall = ragas.get("context_recall", 0)
            avg_ragas = (faithfulness + relevancy + recall) / 3 if all([faithfulness, relevancy, recall]) else 0
            
            writer.writerow({
                'Variant': variant['variant'].upper(),
                'Total Entries': variant['total_entries'],
                'Overall Status': variant['overall_status'],
                'Hallucination Count': variant['total_hallucinations'],
                'Hallucination Rate (%)': f"{variant['hallucination_rate']*100:.1f}%",
                'Verified Count': variant['total_verified'],
                'Unsubstantiated Count': variant['total_unsubstantiated'],
                'Faithfulness': f"{faithfulness:.3f}",
                'Answer Relevancy': f"{relevancy:.3f}",
                'Context Recall': f"{recall:.3f}",
                'Avg RAGAS Score': f"{avg_ragas:.3f}"
            })
    
    print(f"✅ CSV report generated: {csv_path}")
    
    # Print to console as well
    print("\n" + "="*120)
    print("📊 AUDIT SUMMARY TABLE")
    print("="*120)
    
    # Read and display CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Print header
        header = " | ".join(f"{name:^18}" for name in fieldnames)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in rows:
            values = [row[field] for field in fieldnames]
            line = " | ".join(f"{val:^18}" for val in values)
            print(line)
    
    print("="*120)
    print(f"\n📁 Files saved to: {csv_path.parent}/")

if __name__ == "__main__":
    generate_audit_csv()
