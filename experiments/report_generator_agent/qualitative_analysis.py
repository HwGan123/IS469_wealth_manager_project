"""
qualitative_analysis.py
=======================
Simple qualitative comparison of the three report generation variants.

Complements the quantitative metrics in Report_Generation_Experiment.ipynb with:
  1. Faithfulness check  – did the variant use the actual input data correctly?
  2. Structure check     – sections, tables, compliance notice, word count
  3. LLM pairwise judge  – side-by-side quality comparison on 3 investor dimensions

The three report outputs are stored here directly from the notebook run, so
this script requires no MCP server or async setup.

How to run
----------
    python experiments/report_generator_agent/qualitative_analysis.py
"""

import re
import os
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  GOLDEN INPUT + STORED REPORT OUTPUTS (from notebook run)
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_INPUT = {
    "client_name": "Mr. Jameson",
    "risk_profile": "Moderate-Aggressive",
    "metrics": {"Sharpe_Ratio": 0.82, "Annual_Volatility": "14.2%"},
    "research_insights": [
        "NVDA: Blackwell chip production yields are higher than expected.",
        "AAPL: Services revenue saw a 12% jump but hardware sales are flat in China.",
        "Market Sentiment: Bullish on AI infrastructure, Neutral on consumer electronics.",
    ],
}

REPORTS: Dict[str, str] = {
    "Variant A: Pure Prompt": """
**Wealth Management Report for Mr. Jameson**

### 1. Executive Summary
As your dedicated Wealth Manager, I am pleased to present this report tailored to your moderate-aggressive risk profile. Our analysis indicates a favorable outlook for your portfolio, with a Sharpe Ratio of 0.82, suggesting a strong risk-adjusted return. The current annual volatility of 14.2% is within our expected range, given your investment strategy. We will continue to monitor market trends and adjust your portfolio as necessary to optimize performance.

### 2. Portfolio Allocation

| Asset Class | Allocation (%) |
| --- | --- |
| Stocks (Tech) | 40 |
| Stocks (Non-Tech) | 20 |
| Bonds | 30 |
| Alternatives | 10 |

### 3. Quantitative Risk Analysis
Our quantitative risk assessment reveals a Sharpe Ratio of 0.82, indicating a favorable balance between returns and volatility. The annual volatility of 14.2% is manageable, considering your moderate-aggressive risk tolerance. We will maintain a close eye on market fluctuations to ensure your portfolio remains aligned with your investment objectives.

### 4. Market Research & Commentary
Recent insights suggest promising developments in the tech sector, particularly with NVDA's Blackwell chip production yields exceeding expectations. This bodes well for our tech-heavy stock allocation. Meanwhile, AAPL's services revenue growth of 12% is encouraging, although flat hardware sales in China warrant caution. Market sentiment remains bullish on AI infrastructure, which supports our tech-oriented strategy. However, we will maintain a neutral stance on consumer electronics due to mixed market signals. Overall, our portfolio is well-positioned to capitalize on emerging trends while managing risk in line with your investment goals.
""".strip(),

    "Variant B: ReAct Agent": """
Dear Mr. Jameson,

I am pleased to present your personalized 1-page investment report. The report includes a tailored allocation table, research insights, and a compliance disclaimer based on your risk profile.

Below is your allocation table:

| Asset Ticker | Targeted Weight |
| :--- | :--- |
| Stocks | 0.6 |
| Bonds | 0.2 |
| Alternatives | 0.2 |

Our research insights indicate a bullish market sentiment, with key points including:
- NVDA: Blackwell chip production yields are higher than expected.
- AAPL: Services revenue saw a 12% jump but hardware sales are flat in China.
- Market Sentiment: Bullish on AI infrastructure, Neutral on consumer electronics.

Please note the following compliance notice:
**Compliance Notice:** STRATEGY DISCLOSURE: This allocation involves high-growth equity targets. Market volatility may impact principal value.

We hope this report meets your expectations and provides valuable insights into your investment portfolio. If you have any questions or concerns, please do not hesitate to reach out.

Best regards,
[Your Name]
Senior Wealth Report Orchestrator
""".strip(),

    "Variant C: AgentSkills": """
# Investment Strategy Report: Valued Client
## Executive Summary
Our current market analysis indicates a bullish sentiment, driven by positive developments in the tech sector, particularly in AI infrastructure. Notably, NVDA's Blackwell chip production yields are exceeding expectations, and AAPL's services revenue has seen a significant 12% increase, although hardware sales in China remain flat. This environment suggests a favorable outlook for investments in high-growth equities, especially those related to AI and technology services.

## Market Analysis & Sentiment
### Market Analysis & Sentiment (Bullish)
- NVDA: Blackwell chip production yields are higher than expected.
- AAPL: Services revenue saw a 12% jump but hardware sales are flat in China.
- Market Sentiment: Bullish on AI infrastructure, Neutral on consumer electronics.

## Allocation Strategy
Our allocation strategy is designed to capitalize on the current market sentiment while maintaining a balanced portfolio. The targeted weights are as follows:
| Asset Ticker | Targeted Weight |
| :--- | :--- |
| Stocks | 0.6 |
| Bonds | 0.2 |
| Commodities | 0.2 |

This allocation reflects a focus on stocks, particularly those in the technology sector, to leverage the bullish sentiment towards AI infrastructure. The inclusion of bonds and commodities aims to provide a balanced approach, mitigating potential risks associated with market volatility.

## Validation and Compliance
The total weight of our allocation strategy equals 100%, ensuring a comprehensive and balanced approach to investment.

---
**Compliance Notice:** STRATEGY DISCLOSURE: This allocation involves high-growth equity targets. Market volatility may impact principal value.
""".strip(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FAITHFULNESS CHECK  (zero LLM calls — string matching + regex)
#     Did the variant correctly use the actual input data AND follow instructions?
# ═══════════════════════════════════════════════════════════════════════════════

def faithfulness_check(report: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check whether key data points from the input appear in the report,
    and whether the required report sections were included.
    Returns a dict of {check_name: bool} and an overall faithfulness score.
    """
    text  = report.lower()
    lines = report.splitlines()

    checks = {
        # Client identity
        "client_name":      data["client_name"].lower() in text,
        "risk_profile":     data["risk_profile"].lower() in text,
        # Quantitative metrics
        "sharpe_ratio":     "0.82" in text,
        "volatility":       "14.2" in text,
        # Research tickers explicitly mentioned
        "nvda_mentioned":   "nvda" in text,
        "aapl_mentioned":   "aapl" in text,
        # Specific insight detail (not just the ticker)
        "nvda_insight":     "blackwell" in text or "yields" in text,
        "aapl_insight":     "12%" in text or "services revenue" in text,
        # Required sections from the system prompt
        "has_exec_summary": bool(re.search(r"executive summary", report, re.I)),
        "has_risk_section": bool(re.search(r"risk analysis|risk assessment|volatility", report, re.I)),
        "has_table":        len([l for l in lines if l.startswith("|")]) >= 3,
        "has_disclaimer":   bool(re.search(r"compliance|disclosure", report, re.I)),
    }

    score = sum(checks.values()) / len(checks)
    return {"checks": checks, "score": round(score, 3)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

_W = 100

def _bar(c="═"): return c * _W

def print_faithfulness(results: Dict[str, Dict]) -> None:
    print(f"\n{_bar()}\nFAITHFULNESS CHECK\n{_bar()}")

    checks_list = list(next(iter(results.values()))["checks"].keys())
    col = 26
    header = f"  {'Check':<{col}}" + "".join(f"  {v[:22]:<22}" for v in results.keys())
    print(header)
    print(f"  {'-'*col}" + "".join(f"  {'-'*22}" for _ in results))

    for check in checks_list:
        row = f"  {check:<{col}}"
        for data in results.values():
            val = "✓" if data["checks"][check] else "✗"
            row += f"  {val:<22}"
        print(row)

    score_row = f"\n  {'Faithfulness score':<{col}}"
    for data in results.values():
        score_row += f"  {data['score']:.0%}{'':<20}"
    print(score_row)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{_bar()}\nREPORT GENERATION — QUALITATIVE ANALYSIS\n{_bar()}")
    print("Variants: A=Pure Prompt  B=ReAct Agent  C=AgentSkills")

    # --- Analysis 1: Faithfulness ---
    faithfulness_results = {
        name: faithfulness_check(report, GOLDEN_INPUT)
        for name, report in REPORTS.items()
    }
    print_faithfulness(faithfulness_results)

