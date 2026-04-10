#!/usr/bin/env python3
"""
Test script for enhanced Auditor Agent with RAGAS metrics.

This demonstrates:
1. GPT-4o-mini powered auditing
2. RAGAS metrics calculation (faithfulness, answer_relevancy, context_recall)
3. Claim extraction from Investment Analyst output
4. Cross-checking claims against 10-K context
5. Hallucination detection
6. Comprehensive audit output
"""

import os
import json
from dotenv import load_dotenv
from agents.auditor import AuditorAgent, RAGASCalculator

load_dotenv()

def demo_auditor():
    """Demonstrate the enhanced auditor with sample data."""
    
    # Sample investment draft from Investment Analyst
    investment_draft = """
    ## Apple Inc. Investment Analysis
    
    ### Financial Overview
    Apple reported revenue of $394.3 billion in fiscal 2024, representing a 2% year-over-year growth.
    The company maintains a strong cash position of $157 billion, up 10% from prior year.
    
    ### Key Metrics
    - Operating margin: 30.1%
    - Net profit margin: 26.4%
    - Debt-to-equity ratio: 0.15
    - Return on equity: 95%
    
    ### Risk Assessment
    The company faces no significant risks in supply chain management.
    iPhone sales are guaranteed to remain stable with zero volatility.
    
    ### Sentiment
    Market sentiment suggests Apple will always outperform competitors with certainty.
    """
    
    # Sample 10-K context
    context = """
    APPLE INC. - FORM 10-K FISCAL 2024
    
    Net Sales and Revenue:
    Apple reported net sales of $394.3 billion in fiscal 2024, compared to $383.3 billion in fiscal 2023.
    This represents an increase of 2.9% year-over-year.
    
    Cash and Cash Equivalents:
    Cash and cash equivalents were $157.17 billion at September 28, 2024, compared to $143.6 billion 
    at September 30, 2023, representing an 9.4% increase.
    
    Operating Income and Margins:
    Operating income was $123.5 billion with an operating margin of 31.3%.
    Net income was $93.7 billion with a net profit margin of 23.8%.
    
    Financial Leverage:
    Total debt: $97.8 billion
    Total equity: $674.2 billion
    Debt-to-equity ratio: 0.145
    
    Supply Chain Risks:
    The company faces potential supply chain disruptions due to:
    - Concentrated manufacturing in Asia Pacific regions
    - Geopolitical tensions affecting component sourcing
    - Energy transition dependencies
    
    iPhone Revenue:
    iPhone sales increased 5.7% year-over-year, but growth rates vary by quarter
    with potential for negative growth in specific periods.
    """
    
    print("=" * 80)
    print("ENHANCED AUDITOR AGENT - TEST DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize auditor with GPT-4o-mini
        auditor = AuditorAgent()
        
        print("\n📋 INVESTMENT DRAFT:")
        print("-" * 80)
        print(investment_draft)
        
        print("\n📄 RETRIEVED 10-K CONTEXT:")
        print("-" * 80)
        print(context[:500] + "...")
        
        print("\n🔍 RUNNING COMPREHENSIVE AUDIT...")
        print("-" * 80)
        
        # Perform audit
        report = auditor.audit_draft(
            draft=investment_draft,
            context=context,
            ground_truth="Apple 2024 financial results"
        )
        
        # Display results
        print("\n✅ AUDIT REPORT")
        print("-" * 80)
        print(f"Status: {report.status}")
        print(f"Overall Score: {'APPROVED' if report.status == 'APPROVED' else 'REJECTED'}")
        print()
        
        print("📊 CLAIM VERIFICATION RESULTS:")
        print(f"  • Verified Claims: {report.verified_count}")
        print(f"  • Unsubstantiated Claims: {report.unsubstantiated_count}")
        print(f"  • Hallucinations Detected: {report.hallucination_count}")
        print()
        
        print("🔬 RAGAS METRICS:")
        print(f"  • Faithfulness Score: {report.ragas_metrics.faithfulness:.1%}")
        print(f"    (How faithful is the answer to the context?)")
        print(f"  • Answer Relevancy: {report.ragas_metrics.answer_relevancy:.1%}")
        print(f"    (How relevant is the answer to the question?)")
        print(f"  • Context Recall: {report.ragas_metrics.context_recall:.1%}")
        print(f"    (How much of the relevant information was retrieved?)")
        print()
        
        print("🔎 DETAILED FINDINGS:")
        print("-" * 80)
        for i, finding in enumerate(report.findings, 1):
            print(f"\n{i}. Claim: {finding.claim}")
            print(f"   Status: {finding.status}")
            print(f"   Evidence Strength: {finding.evidence_strength}")
            print(f"   Source Quote: {finding.source_quote[:100]}...")
            if finding.correction:
                print(f"   Correction: {finding.correction}")
        
        print("\n" + "=" * 80)
        print("📝 SUMMARY NOTES:")
        print("-" * 80)
        print(report.summary_notes)
        print("=" * 80)
        
        # Output as JSON for programmatic use
        print("\n💾 JSON OUTPUT (for integration):")
        print("-" * 80)
        output = {
            "status": report.status,
            "hallucination_count": report.hallucination_count,
            "verified_count": report.verified_count,
            "unsubstantiated_count": report.unsubstantiated_count,
            "ragas_metrics": {
                "faithfulness": report.ragas_metrics.faithfulness,
                "answer_relevancy": report.ragas_metrics.answer_relevancy,
                "context_recall": report.ragas_metrics.context_recall
            },
            "findings": [
                {
                    "claim": f.claim,
                    "status": f.status,
                    "evidence_strength": f.evidence_strength,
                    "source_quote": f.source_quote,
                    "correction": f.correction
                }
                for f in report.findings
            ]
        }
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(f"❌ Error running audit: {e}")
        import traceback
        traceback.print_exc()

def demo_ragas_calculator():
    """Demonstrate RAGAS metrics calculation."""
    print("\n" + "=" * 80)
    print("RAGAS METRICS CALCULATOR - DIRECT DEMO")
    print("=" * 80)
    
    calc = RAGASCalculator()
    
    # Test 1: Faithfulness
    answer = "Apple reported revenue of $400 billion"
    context = "Apple reported revenue of $394.3 billion in fiscal 2024"
    faithfulness = calc.calculate_faithfulness(answer, context)
    print(f"\n1️⃣ FAITHFULNESS TEST")
    print(f"   Answer: {answer}")
    print(f"   Context: {context}")
    print(f"   Score: {faithfulness:.1%}")
    
    # Test 2: Answer Relevancy
    question = "What is Apple's revenue?"
    answer = "Apple's total revenue for 2024 was $394.3 billion"
    relevancy = calc.calculate_answer_relevancy(answer, question)
    print(f"\n2️⃣ ANSWER RELEVANCY TEST")
    print(f"   Question: {question}")
    print(f"   Answer: {answer}")
    print(f"   Score: {relevancy:.1%}")
    
    # Test 3: Context Recall
    ground_truth = "Apple's debt-to-equity ratio is below 0.15"
    context = "Total debt: $97.8 billion, Total equity: $674.2 billion, Ratio: 0.145"
    recall = calc.calculate_context_recall(context, ground_truth)
    print(f"\n3️⃣ CONTEXT RECALL TEST")
    print(f"   Ground Truth: {ground_truth}")
    print(f"   Retrieved Context: {context}")
    print(f"   Score: {recall:.1%}")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        exit(1)
    
    # Run demonstrations
    demo_auditor()
    demo_ragas_calculator()
    
    print("\n✅ Test complete!")
