# agents/metrics_judge.py
import json
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class MetricsJudge:
    def __init__(self, api_key: str = None):
        # We use a high-reasoning model (70B) for the Judge to ensure it is 
        # "smarter" than the models it is grading.
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0, 
            api_key=api_key
        )

    def evaluate(self, report: str, context: str) -> Dict:
        """
        Grades a report based on Factual Grounding, Hallucination Rate, and Quality.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an independent AI Audit Specialist. 
            Your task is to provide a quantitative evaluation of a financial report 
            based ONLY on the provided context (10-K data)."""),
            ("user", """
            ### REFERENCE CONTEXT (10-K DATA):
            {context}

            ### AGENT GENERATED REPORT:
            {report}

            ### EVALUATION RUBRIC:
            1. **Hallucination Rate (0.0 to 1.0)**: 
               - 0.0: Every single number and claim is found in the context.
               - 1.0: All numbers and claims are fabricated.
               - Calculation: (Count of claims NOT in context) / (Total number of claims).

            2. **Factual Grounding (0.0 to 1.0)**:
               - Measures the depth of usage of the context. 
               - 1.0: Extensively uses specific quotes and data points from the context.
               - 0.0: Uses generic "training data" knowledge with no specific context references.

            3. **Quality Score (1 to 10)**:
               - 10: Professional, logically sound, and follows all financial formatting.
               - 1: Unprofessional, contradictory, or structurally messy.

            ### OUTPUT FORMAT:
            Return ONLY a valid JSON object with these keys:
            {{
                "hallucination_rate": float,
                "factual_grounding": float,
                "quality_score": int,
                "reasoning": "string explanation of the scores"
            }}
            """)
        ])

        # Execute the judge call
        chain = prompt | self.llm
        response = chain.invoke({"report": report, "context": context})

        try:
            # Clean the response content in case the LLM adds markdown triple backticks
            clean_content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_content)
        except Exception as e:
            print(f"Error parsing Judge output: {e}")
            return {{
                "hallucination_rate": 1.0, 
                "factual_grounding": 0.0, 
                "quality_score": 0, 
                "reasoning": "Error: Judge failed to produce valid JSON."
            }}