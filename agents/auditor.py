# agents/auditor.py
import os
import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from graph.state import WealthManagerState

# 1. Define the Structured Output Schema
class FactCheck(BaseModel):
    claim: str = Field(description="The specific financial fact or number from the draft.")
    status: Literal["VERIFIED", "HALLUCINATION", "UNSUBSTANTIATED"] = Field(description="Status of the claim.")
    source_quote: str = Field(description="The exact quote from the 10-K context that supports or refutes this.")
    correction: str = Field(description="If hallucinated, provide the correct number from the context.")

class AuditReport(BaseModel):
    status: Literal["APPROVED", "REJECTED"] = Field(description="Final verdict of the audit.")
    findings: List[FactCheck] = Field(description="List of all claims checked.")
    summary_notes: str = Field(description="Overall feedback for the Investment Agent.")

# 2. Define the Auditor Agent Class
class AuditorAgent:
    def __init__(self, api_key: str):
        # Temperature 0 is critical for auditing to prevent the judge from hallucinating
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
        self.parser = PydanticOutputParser(pydantic_object=AuditReport)

    def audit_draft(self, draft: str, context: str) -> AuditReport:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Compliance Auditor for a Tier-1 Wealth Manager. 
            Your success is measured by your ability to find errors. 
            Do NOT trust the Investment Agent. Cross-reference every number.
            
            {format_instructions}"""),
            ("user", """
            INVESTMENT DRAFT:
            {draft}
            
            GOLDEN SOURCE (10-K CONTEXT):
            {context}
            
            Perform a line-by-line audit. Check for:
            1. Numerical accuracy (Revenue, Debt, Growth rates).
            2. Logic gaps (e.g., claiming high growth when risks state supply chain issues).
            3. Sentiment alignment.
            """)
        ])

        # Inject formatting instructions for the Pydantic parser
        input_data = prompt.format_prompt(
            draft=draft, 
            context=context, 
            format_instructions=self.parser.get_format_instructions()
        )
        
        response = self.llm.invoke(input_data.to_messages())
        
        try:
            return self.parser.parse(response.content)
        except Exception as e:
            # Fallback in case of parsing errors
            return AuditReport(
                status="REJECTED", 
                findings=[], 
                summary_notes=f"Audit failed due to parsing error: {str(e)}"
            )


def _fallback_audit(draft: str, context: str) -> dict:
    draft_text = (draft or "").lower()
    context_text = (context or "").lower()

    hallucination_hits = 0
    for token in ["guaranteed", "certain", "risk-free"]:
        if token in draft_text:
            hallucination_hits += 1

    has_context_overlap = bool(context_text and any(word in context_text for word in ["risk", "revenue", "guidance", "cash"]))
    is_hallucinating = hallucination_hits > 0 or not has_context_overlap
    score = 0.35 if is_hallucinating else 0.9

    notes = (
        "Draft includes unsupported certainty-style language or weak grounding."
        if is_hallucinating
        else "Draft appears reasonably grounded against available context."
    )
    return {
        "audit_score": score,
        "is_hallucinating": is_hallucinating,
        "audit_findings": [{"summary": notes}],
        "messages": [f"Auditor: {'REJECTED' if is_hallucinating else 'APPROVED'} (score={score:.2f})"],
    }


def auditor_node(state: WealthManagerState) -> dict:
    print("--- AGENT: AUDITOR ---")

    draft = state.get("draft_report", "")
    context = state.get("retrieved_context", "")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_audit(draft, context)

    try:
        agent = AuditorAgent(api_key=api_key)
        report = agent.audit_draft(draft=draft, context=context)
        is_hallucinating = report.status != "APPROVED"
        score = 0.35 if is_hallucinating else 0.9

        return {
            "audit_score": score,
            "is_hallucinating": is_hallucinating,
            "audit_findings": [f.model_dump() for f in report.findings],
            "messages": [f"Auditor: {report.status} (score={score:.2f})"],
        }
    except Exception:
        return _fallback_audit(draft, context)