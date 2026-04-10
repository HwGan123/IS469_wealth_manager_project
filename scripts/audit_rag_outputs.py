#!/usr/bin/env python3
"""
RAG Output Auditor - Comprehensive Claim Verification & RAGAS Analysis

Reads qualitative RAG outputs (.jsonl files) and runs the enhanced auditor
against each generated answer, producing:
1. Claim verification list
2. Hallucination count per variant
3. RAGAS scores (faithfulness, answer_relevancy, context_recall)
4. Comparative audit report across all variants
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.auditor import AuditorAgent, RAGASCalculator
from dotenv import load_dotenv

load_dotenv()

class RAGAuditor:
    """Audit RAG outputs with comprehensive claim verification."""
    
    def __init__(self):
        self.auditor = AuditorAgent()
        self.results_dir = Path("results")
        self.audit_reports = {}
        
    def load_jsonl_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load .jsonl file (one JSON per line)."""
        entries = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            return entries
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return []
    
    def format_contexts(self, contexts: List[str]) -> str:
        """Combine multiple context chunks into single string."""
        return "\n\n---\n\n".join(contexts[:5])  # Limit to first 5 chunks
    
    def audit_rag_output(self, question: str, answer: str, contexts: List[str], 
                         ground_truth: str) -> Dict[str, Any]:
        """Audit a single RAG output (question-answer-context triple)."""
        # Combine contexts
        context_str = self.format_contexts(contexts)
        
        try:
            # Run comprehensive audit
            report = self.auditor.audit_draft(
                draft=answer,
                context=context_str,
                ground_truth=ground_truth
            )
            
            return {
                "status": report.status,
                "hallucination_count": report.hallucination_count,
                "verified_count": report.verified_count,
                "unsubstantiated_count": report.unsubstantiated_count,
                "ragas_metrics": {
                    "faithfulness": report.ragas_metrics.faithfulness,
                    "answer_relevancy": report.ragas_metrics.answer_relevancy,
                    "context_recall": report.ragas_metrics.context_recall
                },
                "findings_summary": f"{report.hallucination_count} hallucinations, "
                                   f"{report.verified_count} verified, "
                                   f"{report.unsubstantiated_count} unsubstantiated"
            }
        except Exception as e:
            print(f"  ⚠️  Error auditing question: {str(e)[:50]}")
            return {
                "status": "ERROR",
                "error": str(e),
                "hallucination_count": 0,
                "verified_count": 0,
                "unsubstantiated_count": 0,
                "ragas_metrics": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_recall": 0.0
                }
            }
    
    def audit_variant(self, variant_name: str, jsonl_path: Path) -> Dict[str, Any]:
        """Audit all entries in a variant's .jsonl file."""
        print(f"\n📊 Auditing variant: {variant_name}")
        print("-" * 80)
        
        entries = self.load_jsonl_file(jsonl_path)
        if not entries:
            print(f"  ❌ No entries found in {jsonl_path}")
            return None
        
        audit_results = []
        total_hallucinations = 0
        total_verified = 0
        total_unsubstantiated = 0
        faithfulness_scores = []
        relevancy_scores = []
        recall_scores = []
        
        for i, entry in enumerate(entries, 1):
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            contexts = entry.get("contexts", [])
            ground_truth = entry.get("ground_truth", "")
            
            # Run audit
            audit = self.audit_rag_output(question, answer, contexts, ground_truth)
            audit["question_id"] = entry.get("id", i)
            audit["question"] = question
            
            audit_results.append(audit)
            
            # Aggregate metrics
            if audit.get("status") != "ERROR":
                total_hallucinations += audit.get("hallucination_count", 0)
                total_verified += audit.get("verified_count", 0)
                total_unsubstantiated += audit.get("unsubstantiated_count", 0)
                
                metrics = audit.get("ragas_metrics", {})
                faithfulness_scores.append(metrics.get("faithfulness", 0))
                relevancy_scores.append(metrics.get("answer_relevancy", 0))
                recall_scores.append(metrics.get("context_recall", 0))
            
            # Progress indicator
            if i % 5 == 0:
                print(f"  ✓ Audited {i}/{len(entries)} entries")
        
        # Calculate averages
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        
        # Determine overall status
        hallucination_rate = total_hallucinations / len(entries) if entries else 0
        overall_status = "PASSED" if (hallucination_rate < 0.2 and avg_faithfulness > 0.7) else "FAILED"
        
        variant_summary = {
            "variant": variant_name,
            "total_entries": len(entries),
            "overall_status": overall_status,
            "hallucination_rate": hallucination_rate,
            "total_hallucinations": total_hallucinations,
            "total_verified": total_verified,
            "total_unsubstantiated": total_unsubstantiated,
            "ragas_averages": {
                "faithfulness": round(avg_faithfulness, 3),
                "answer_relevancy": round(avg_relevancy, 3),
                "context_recall": round(avg_recall, 3)
            },
            "audit_details": audit_results
        }
        
        # Print summary
        print(f"  ✅ Status: {overall_status}")
        print(f"  📈 Hallucination Rate: {hallucination_rate:.1%} ({total_hallucinations}/{len(entries)})")
        print(f"  ✓ Verified: {total_verified}, Unsubstantiated: {total_unsubstantiated}")
        print(f"  🔬 RAGAS Averages:")
        print(f"     - Faithfulness: {avg_faithfulness:.1%}")
        print(f"     - Answer Relevancy: {avg_relevancy:.1%}")
        print(f"     - Context Recall: {avg_recall:.1%}")
        
        return variant_summary
    
    def audit_all_variants(self, base_dir: str = "results/rag_compare") -> Dict[str, Any]:
        """Audit all RAG variants in a directory."""
        base_path = Path(base_dir)
        
        if not base_path.exists():
            print(f"❌ Directory not found: {base_dir}")
            return {}
        
        variants_to_audit = []
        
        # Find all .jsonl files
        for jsonl_file in base_path.rglob("*_details.jsonl"):
            variant_name = jsonl_file.stem.replace("_details", "")
            variants_to_audit.append((variant_name, jsonl_file))
        
        if not variants_to_audit:
            print(f"❌ No .jsonl files found in {base_dir}")
            return {}
        
        print(f"🔍 Found {len(variants_to_audit)} RAG variant(s) to audit")
        
        all_results = {
            "audit_source": base_dir,
            "variants_audited": [],
            "comparative_summary": {}
        }
        
        for variant_name, jsonl_path in variants_to_audit:
            result = self.audit_variant(variant_name, jsonl_path)
            if result:
                all_results["variants_audited"].append(result)
        
        # Generate comparative summary
        all_results["comparative_summary"] = self.generate_comparative_summary(
            all_results["variants_audited"]
        )
        
        return all_results
    
    def generate_comparative_summary(self, variants: List[Dict]) -> Dict[str, Any]:
        """Generate comparative analysis across variants."""
        if not variants:
            return {}
        
        summary = {
            "best_variant_by_faithfulness": None,
            "best_variant_by_relevancy": None,
            "best_variant_by_recall": None,
            "lowest_hallucination_rate": None,
            "variant_rankings": []
        }
        
        # Find best performers
        best_faith = max(variants, key=lambda v: v["ragas_averages"]["faithfulness"])
        best_relev = max(variants, key=lambda v: v["ragas_averages"]["answer_relevancy"])
        best_rec = max(variants, key=lambda v: v["ragas_averages"]["context_recall"])
        lowest_hall = min(variants, key=lambda v: v["hallucination_rate"])
        
        summary["best_variant_by_faithfulness"] = {
            "variant": best_faith["variant"],
            "score": best_faith["ragas_averages"]["faithfulness"]
        }
        summary["best_variant_by_relevancy"] = {
            "variant": best_relev["variant"],
            "score": best_relev["ragas_averages"]["answer_relevancy"]
        }
        summary["best_variant_by_recall"] = {
            "variant": best_rec["variant"],
            "score": best_rec["ragas_averages"]["context_recall"]
        }
        summary["lowest_hallucination_rate"] = {
            "variant": lowest_hall["variant"],
            "rate": lowest_hall["hallucination_rate"],
            "count": lowest_hall["total_hallucinations"]
        }
        
        # Rankings by RAGAS average
        variant_scores = []
        for v in variants:
            metrics = v["ragas_averages"]
            avg_score = (metrics["faithfulness"] + metrics["answer_relevancy"] + 
                        metrics["context_recall"]) / 3
            variant_scores.append({
                "variant": v["variant"],
                "average_ragas_score": round(avg_score, 3),
                "hallucination_rate": v["hallucination_rate"],
                "status": v["overall_status"]
            })
        
        summary["variant_rankings"] = sorted(
            variant_scores,
            key=lambda x: x["average_ragas_score"],
            reverse=True
        )
        
        return summary
    
    def save_audit_report(self, audit_results: Dict[str, Any], 
                         output_dir: str = "results/auditor_compare.py") -> Path:
        """Save audit report to JSON file."""
        output_path = Path(output_dir) / "rag_audit_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(audit_results, f, indent=2)
        
        print(f"\n💾 Audit report saved to: {output_path}")
        return output_path
    
    def print_final_summary(self, audit_results: Dict[str, Any]):
        """Print formatted audit summary."""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE RAG AUDIT SUMMARY")
        print("=" * 80)
        
        summary = audit_results.get("comparative_summary", {})
        
        if summary.get("best_variant_by_faithfulness"):
            print(f"\n⭐ Best by Faithfulness:")
            best = summary["best_variant_by_faithfulness"]
            print(f"   {best['variant'].upper()}: {best['score']:.1%}")
        
        if summary.get("lowest_hallucination_rate"):
            low = summary["lowest_hallucination_rate"]
            print(f"\n✓ Lowest Hallucination Rate:")
            print(f"   {low['variant'].upper()}: {low['rate']:.1%} "
                  f"({low['count']} total)")
        
        if summary.get("variant_rankings"):
            print(f"\n📊 Overall Rankings (by avg RAGAS score):")
            for i, ranking in enumerate(summary["variant_rankings"], 1):
                status_emoji = "✅" if ranking["status"] == "PASSED" else "❌"
                print(f"   {i}. {ranking['variant'].upper():20} - "
                      f"RAGAS: {ranking['average_ragas_score']:.1%}, "
                      f"Hallucinations: {ranking['hallucination_rate']:.1%} {status_emoji}")
        
        print("\n" + "=" * 80)

def main():
    """Main audit execution."""
    print("🚀 RAG OUTPUT AUDITOR - Starting comprehensive audit")
    print("=" * 80)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    # Run audit
    auditor = RAGAuditor()
    
    # Audit baseline RAG variants
    print("\n🔍 Auditing baseline RAG variants (rag_compare)...")
    audit_results = auditor.audit_all_variants("results/rag_compare")
    
    if audit_results.get("variants_audited"):
        # Save report
        auditor.save_audit_report(audit_results)
        
        # Print summary
        auditor.print_final_summary(audit_results)
        
        print("\n✅ Audit complete!")
    else:
        print("❌ No variants audited. Check results directory.")

if __name__ == "__main__":
    main()
