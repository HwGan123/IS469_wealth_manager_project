from graph.state import WealthManagerState


def report_generator_node(state: WealthManagerState) -> dict:
    print("--- AGENT: REPORT GENERATOR ---")

    tickers = state.get("tickers") or []
    sentiment_score = float(state.get("sentiment_score", 0.0) or 0.0)
    draft = state.get("draft_report", "") or "No draft report was generated."
    audit_score = float(state.get("audit_score", 0.0) or 0.0)
    is_hallucinating = bool(state.get("is_hallucinating", False))

    verdict = "REVIEW REQUIRED" if is_hallucinating else "APPROVED"

    final_report = "\n".join(
        [
            "# Final Wealth Manager Report",
            "",
            "## Agent Summary",
            f"- Tickers: {', '.join(tickers) if tickers else 'N/A'}",
            f"- Sentiment score: {sentiment_score:+.2f}",
            f"- Audit score: {audit_score:.2f}",
            f"- Audit verdict: {verdict}",
            "",
            "## Investment Analysis",
            draft,
        ]
    )

    return {
        "final_report": final_report,
        "messages": [final_report],
    }
