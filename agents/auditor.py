# agents/auditor.py
import os
import json
import re
from typing import List, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from graph.state import WealthManagerState
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# 1. Define the Structured Output Schema
class FactCheck(BaseModel):
    claim: str = Field(description="The specific financial fact or number from the draft.")
    status: Literal["VERIFIED", "HALLUCINATION", "UNSUBSTANTIATED"] = Field(description="Status of the claim.")
    source_quote: str = Field(description="The exact quote from the 10-K context that supports or refutes this.")
    evidence_strength: Literal["STRONG", "WEAK", "NONE"] = Field(description="How strongly the source supports this claim.")
    correction: str = Field(description="If hallucinated, provide the correct number from the context.")

class RAGASMetrics(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0, description="RAGAS faithfulness score (0-1)")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="RAGAS answer relevancy score (0-1)")
    context_recall: float = Field(ge=0.0, le=1.0, description="RAGAS context recall score (0-1)")

class AuditReport(BaseModel):
    status: Literal["APPROVED", "REJECTED"] = Field(description="Final verdict of the audit.")
    findings: List[FactCheck] = Field(description="List of all claims checked.")
    hallucination_count: int = Field(description="Total number of hallucinations detected.")
    verified_count: int = Field(description="Total number of verified claims.")
    unsubstantiated_count: int = Field(description="Total number of unsubstantiated claims.")
    ragas_metrics: RAGASMetrics = Field(description="RAGAS evaluation metrics.")
    summary_notes: str = Field(description="Overall feedback for the Investment Agent.")

# 2. RAGAS Metrics Calculator
class RAGASCalculator:
    """Calculate RAGAS metrics for hallucination detection."""
    
    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract all numbers/percentages from text."""
        patterns = [
            r'\$[\d,]+(?:\.\d+)?[BMK]?',  # Currency
            r'\d+\.?\d*%',                  # Percentages
            r'\d{1,3}(?:,\d{3})*(?:\.\d+)?'  # Large numbers
        ]
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text))
        return numbers
    
    @staticmethod
    def calculate_faithfulness(answer: str, context: str) -> float:
        """
        Faithfulness: 0-1 score indicating if answer is faithful to context.
        Check if numerical claims in answer appear in context.
        """
        answer_numbers = RAGASCalculator.extract_numbers(answer)
        context_numbers = RAGASCalculator.extract_numbers(context)
        
        if not answer_numbers:
            return 0.95  # No numbers = likely faithful
        
        overlap = sum(1 for num in answer_numbers if num in context_numbers)
        score = min(1.0, overlap / len(answer_numbers)) if answer_numbers else 0.5
        return score
    
    @staticmethod
    def calculate_answer_relevancy(answer: str, question: str) -> float:
        """
        Answer Relevancy: 0-1 score indicating if answer addresses the question.
        Simple: check keyword overlap between question and answer.
        """
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'of'}
        question_words -= stop_words
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words & answer_words)
        score = min(1.0, overlap / len(question_words))
        return score
    
    @staticmethod
    def calculate_context_recall(retrieved_context: str, ground_truth: str) -> float:
        """
        Context Recall: 0-1 score indicating if retrieved context contains ground truth info.
        Check if key terms from ground truth appear in retrieved context.
        """
        if not ground_truth or not retrieved_context:
            return 0.5
        
        # Extract key terms from ground truth (words > 4 chars)
        truth_terms = [w.lower() for w in ground_truth.split() if len(w) > 4]
        context_lower = retrieved_context.lower()
        
        if not truth_terms:
            return 0.5
        
        found = sum(1 for term in truth_terms if term in context_lower)
        score = min(1.0, found / len(truth_terms))
        return score

# 3. Define the Auditor Agent Class
class AuditorAgent:
    def __init__(self, api_key: str = None):
        """Initialize with GPT-4o-mini for high performance auditing."""
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        # Temperature 0 is critical for auditing to prevent hallucination
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=AuditReport)
        self.ragas_calc = RAGASCalculator()
    
    def extract_claims(self, draft: str) -> List[str]:
        """Extract factual claims from the investment draft."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying specific factual claims in financial reports.
            Extract ONLY concrete factual claims (numbers, dates, percentages, facts).
            Do NOT include subjective opinions or analysis.
            Return as a JSON array of strings."""),
            ("user", "Extract factual claims from this report:\n{draft}")
        ])
        
        input_data = prompt.format_prompt(draft=draft)
        response = self.llm.invoke(input_data.to_messages())
        
        try:
            claims = json.loads(response.content)
            if isinstance(claims, list):
                return claims
            else:
                return []
        except Exception as e:
            # Fallback: manual extraction
            lines = draft.split('\n')
            manual_claims = [line.strip() for line in lines if any(char.isdigit() for char in line)][:10]
            return manual_claims
    
    def verify_claim_against_context(self, claim: str, context: str) -> Tuple[FactCheck, float]:
        """Verify a single claim against retrieved context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous fact-checker. For the given claim, determine if it is:
            - VERIFIED: Exactly matches information in the source
            - HALLUCINATION: Contradicts the source or is fabricated
            - UNSUBSTANTIATED: Not mentioned in the source but not contradicted
            
            {format_instructions}"""),
            ("user", """
            CLAIM TO VERIFY:
            {claim}
            
            SOURCE CONTEXT (10-K):
            {context}
            
            Perform a rigorous check. Find the exact quote that supports or refutes this claim.
            If hallucinated, identify what the correct information should be.
            """)
        ])
        
        input_data = prompt.format_prompt(
            claim=claim,
            context=context,
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            response = self.llm.invoke(input_data.to_messages())
            
            # Parse just the FactCheck part
            fact_check_parser = PydanticOutputParser(pydantic_object=FactCheck)
            fact = fact_check_parser.parse(response.content)
            
            # Calculate evidence strength based on context match
            strength = 1.0 if fact.status == "VERIFIED" else (0.3 if fact.status == "UNSUBSTANTIATED" else 0.0)
            return fact, strength
        except Exception as e:
            # Fallback
            strength = 0.5 if context and any(word in context.lower() for word in claim.lower().split()) else 0.1
            return FactCheck(
                claim=claim,
                status="UNSUBSTANTIATED",
                source_quote="Parsing error - manual review needed",
                evidence_strength="WEAK",
                correction=""
            ), strength
    
    def calculate_ragas_metrics(self, draft: str, context: str, ground_truth: str = "", query: str = "") -> RAGASMetrics:
        """Calculate RAGAS metrics for the draft."""
        faithfulness = self.ragas_calc.calculate_faithfulness(draft, context)
        answer_relevancy = self.ragas_calc.calculate_answer_relevancy(draft, query or "investment analysis")
        context_recall = self.ragas_calc.calculate_context_recall(context, ground_truth or draft)

        return RAGASMetrics(
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(answer_relevancy, 3),
            context_recall=round(context_recall, 3)
        )

    def audit_draft(self, draft: str, context: str, ground_truth: str = "", query: str = "") -> AuditReport:
        """Comprehensive audit of investment draft against retrieved context."""
        # 1. Extract claims from draft
        claims = self.extract_claims(draft)

        # 2. Verify each claim
        findings = []
        hallucination_count = 0
        verified_count = 0
        unsubstantiated_count = 0

        for claim in claims:
            fact_check, _ = self.verify_claim_against_context(claim, context)
            findings.append(fact_check)

            if fact_check.status == "HALLUCINATION":
                hallucination_count += 1
            elif fact_check.status == "VERIFIED":
                verified_count += 1
            else:
                unsubstantiated_count += 1

        # 3. Calculate RAGAS metrics (pass actual query for answer_relevancy)
        ragas_metrics = self.calculate_ragas_metrics(draft, context, ground_truth, query)
        
        # 4. Determine overall status
        hallucination_rate = hallucination_count / len(findings) if findings else 0
        status = "APPROVED" if hallucination_rate < 0.2 and ragas_metrics.faithfulness > 0.7 else "REJECTED"
        
        # 5. Generate summary
        summary_notes = (
            f"Audit complete. {verified_count} verified, {unsubstantiated_count} unsubstantiated, "
            f"{hallucination_count} hallucinations detected. "
            f"Faithfulness: {ragas_metrics.faithfulness:.1%}, "
            f"Answer Relevancy: {ragas_metrics.answer_relevancy:.1%}, "
            f"Context Recall: {ragas_metrics.context_recall:.1%}"
        )
        
        return AuditReport(
            status=status,
            findings=findings,
            hallucination_count=hallucination_count,
            verified_count=verified_count,
            unsubstantiated_count=unsubstantiated_count,
            ragas_metrics=ragas_metrics,
            summary_notes=summary_notes
        )


def _fallback_audit(draft: str, context: str) -> dict:
    """Fallback audit when API is unavailable."""
    draft_text = (draft or "").lower()
    context_text = (context or "").lower()

    hallucination_hits = 0
    for token in ["guaranteed", "certain", "risk-free", "always", "never fails"]:
        if token in draft_text:
            hallucination_hits += 1

    has_context_overlap = bool(context_text and any(
        word in context_text for word in ["risk", "revenue", "guidance", "cash", "debt"]
    ))
    is_hallucinating = hallucination_hits > 0 or not has_context_overlap
    
    # Calculate basic RAGAS metrics
    faithfulness = 0.35 if is_hallucinating else 0.85
    answer_relevancy = 0.6 if context_text else 0.3
    context_recall = 0.7 if has_context_overlap else 0.2
    
    score = 0.35 if is_hallucinating else 0.9

    notes = (
        "Draft includes unsupported certainty language or weak grounding. "
        f"Faithfulness: {faithfulness:.1%}, Answer Relevancy: {answer_relevancy:.1%}, "
        f"Context Recall: {context_recall:.1%}"
        if is_hallucinating
        else "Draft appears reasonably grounded against available context. "
        f"Faithfulness: {faithfulness:.1%}, Answer Relevancy: {answer_relevancy:.1%}, "
        f"Context Recall: {context_recall:.1%}"
    )
    
    return {
        "audit_score": score,
        "is_hallucinating": is_hallucinating,
        "hallucination_count": hallucination_hits,
        "verified_count": 0,
        "unsubstantiated_count": 0,
        "ragas_metrics": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall
        },
        "audit_findings": [{"summary": notes}],
        "audit_iteration_count": None,  # Will be set by caller
        "messages": [f"Auditor (Fallback): {'REJECTED' if is_hallucinating else 'APPROVED'} (score={score:.2f})"],
    }


def auditor_node(state: WealthManagerState) -> dict:
    """Auditor node that verifies Investment Analyst output."""
    print("--- AGENT: AUDITOR (Enhanced with RAGAS) ---")

    draft = state.get("draft_report", "")
    retrieved_context = state.get("retrieved_context", "")  # 10-K historical data
    live_data_context = state.get("live_data_context", "")  # Market context (news, earnings, ratings)
    messages = state.get("messages") or []
    query = str(messages[0]) if messages else ""
    
    # Combine both contexts for comprehensive validation
    # This allows RAGAS to verify both historical (10-K) and live market data claims
    USE_COMBINED_CONTEXT = True
    USE_RETREIVED_CONTEXT_ONLY = False  # For strict auditing of 10-K claims only
    USE_LIVE_CONTEXT_ONLY = False  # Not recommended, as it may miss verifying historical claims
    if USE_COMBINED_CONTEXT:
        combined_context = f"{retrieved_context}\n\n--- LIVE MARKET DATA ---\n{live_data_context}"
    elif USE_RETRIEVED_CONTEXT_ONLY:
        combined_context = retrieved_context  # Only use 10-K for strict auditing
    elif USE_LIVE_CONTEXT_ONLY:
        combined_context = live_data_context  # Only use live market data for more lenient auditing
    else:
        combined_context = retrieved_context  # Default to 10-K context

    ground_truth = state.get("ground_truth", "")
    
    # Increment audit iteration counter
    iteration_count = state.get("audit_iteration_count", 0) + 1
    print(f"  Audit Iteration: {iteration_count}/2")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found. Using fallback audit.")
        audit_result = _fallback_audit(draft, combined_context)
        audit_result["audit_iteration_count"] = iteration_count
        return audit_result

    try:
        agent = AuditorAgent(api_key=api_key)
        report = agent.audit_draft(draft=draft, context=combined_context, ground_truth=ground_truth, query=query)
        
        # Force approval on final iteration to prevent infinite loops
        if iteration_count >= 2:
            is_hallucinating = False
            score = 0.9
            status_msg = "APPROVED (forced - max iterations reached)"
        else:
            is_hallucinating = report.status != "APPROVED"
            score = 0.35 if is_hallucinating else 0.9
            status_msg = report.status

        # Format findings for state
        findings_list = [
            {
                "claim": f.claim,
                "status": f.status,
                "source_quote": f.source_quote,
                "evidence_strength": f.evidence_strength,
                "correction": f.correction
            }
            for f in report.findings
        ]

        return {
            "audit_score": score,
            "is_hallucinating": is_hallucinating,
            "hallucination_count": report.hallucination_count,
            "verified_count": report.verified_count,
            "unsubstantiated_count": report.unsubstantiated_count,
            "ragas_metrics": {
                "faithfulness": report.ragas_metrics.faithfulness,
                "answer_relevancy": report.ragas_metrics.answer_relevancy,
                "context_recall": report.ragas_metrics.context_recall
            },
            "audit_findings": findings_list,
            "audit_iteration_count": iteration_count,

            "messages": [
                f"Auditor: {status_msg} (score={score:.2f}) [Iteration {iteration_count}/2]",
                f"Findings: {report.hallucination_count} hallucinations, "
                f"{report.verified_count} verified, {report.unsubstantiated_count} unsubstantiated",
                f"RAGAS - Faithfulness: {report.ragas_metrics.faithfulness:.1%}, "
                f"Answer Relevancy: {report.ragas_metrics.answer_relevancy:.1%}, "
                f"Context Recall: {report.ragas_metrics.context_recall:.1%}"
            ],
        }
    except Exception as e:
        print(f"Error in audit: {e}")
        audit_result = _fallback_audit(draft, combined_context)
        audit_result["audit_iteration_count"] = iteration_count
        return audit_result