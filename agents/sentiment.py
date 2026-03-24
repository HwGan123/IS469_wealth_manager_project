import os
import requests
from langchain_core.tools import tool
from graph.state import WealthManagerState


# ── Tool 1: News Fetcher ───────────────────────────────────────────────────────

@tool
def fetch_ticker_news(tickers: list[str], max_per_ticker: int = 5) -> list[dict]:
    """
    Fetch recent English news headlines from NewsAPI for each ticker.
    Returns a flat list of article dicts with keys:
      ticker, title, description, url, published_at, source
    """
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        raise ValueError("NEWS_API_KEY is not set. Add it to your .env file.")

    articles = []
    for ticker in tickers:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": max_per_ticker,
                    "apiKey": api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            for a in resp.json().get("articles", []):
                articles.append({
                    "ticker":       ticker,
                    "title":        a.get("title") or "",
                    "description":  a.get("description") or "",
                    "url":          a.get("url") or "",
                    "published_at": a.get("publishedAt") or "",
                    "source":       (a.get("source") or {}).get("name") or "",
                })
        except requests.RequestException as e:
            articles.append({
                "ticker": ticker, "title": f"[ERROR fetching news: {e}]",
                "description": "", "url": "", "published_at": "", "source": "",
            })

    return articles


# ── Placeholder / real inference backend ──────────────────────────────────────

def _placeholder_inference(text: str) -> dict:
    """
    Placeholder model — returns uniform scores.
    Swap this out for FinBERT / LLaMA / FinGPT when available.
    """
    return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}


def _finbert_inference(text: str) -> dict:
    """FinBERT inference, loaded lazily and cached on first call."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not hasattr(_finbert_inference, "_tok"):
        _finbert_inference._tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_inference._mdl = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )

    inputs = _finbert_inference._tok(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        probs = (
            torch.nn.functional.softmax(
                _finbert_inference._mdl(**inputs).logits, dim=-1
            )
            .numpy()[0]
        )
    # FinBERT: 0=positive, 1=negative, 2=neutral
    return {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral":  float(probs[2]),
    }


# Swap backend via env var: SENTIMENT_MODEL=finbert  (default: placeholder)
_BACKEND = os.environ.get("SENTIMENT_MODEL", "placeholder").lower()
_infer = _finbert_inference if _BACKEND == "finbert" else _placeholder_inference


# ── Tool 2: Sentiment Inference ────────────────────────────────────────────────

@tool
def run_sentiment_inference(articles: list[dict]) -> list[dict]:
    """
    Run sentiment inference on each article's title (fallback: description).
    Annotates each article with:
      scores   — {positive: float, negative: float, neutral: float}
      label    — dominant sentiment ("positive" | "negative" | "neutral")
    """
    results = []
    for article in articles:
        text = article.get("title") or article.get("description") or ""
        if not text or text.startswith("[ERROR"):
            results.append({**article, "scores": {}, "label": "unknown"})
            continue

        scores = _infer(text)
        label  = max(scores, key=lambda k: scores[k])
        results.append({**article, "scores": scores, "label": label})

    return results


# ── Aggregation helper ─────────────────────────────────────────────────────────

def _aggregate(results: list[dict]) -> dict:
    """
    Compute per-ticker and overall aggregated sentiment.
    Returns:
      overall   — {positive: int, negative: int, neutral: int, score: float}
      by_ticker — {TICKER: {positive: int, negative: int, neutral: int, score: float}}
      summary   — human-readable string
    """
    from collections import defaultdict

    by_ticker: dict = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "score": 0.0})
    overall: dict = {"positive": 0, "negative": 0, "neutral": 0, "score": 0.0}

    for r in results:
        label  = r.get("label", "neutral")
        ticker = r.get("ticker", "UNKNOWN")
        if label in overall:
            overall[label]         += 1
            by_ticker[ticker][label] += 1

    def _score(counts: dict) -> float:
        total = sum(counts.values()) or 1
        return round((counts["positive"] - counts["negative"]) / total, 4)

    overall["score"] = float(_score(overall))
    for t in by_ticker:
        by_ticker[t]["score"] = float(_score(by_ticker[t]))

    # Human-readable summary
    lines = ["## Sentiment Summary\n"]
    for ticker, counts in by_ticker.items():
        lines.append(
            f"**{ticker}**: {counts['positive']} positive, "
            f"{counts['negative']} negative, {counts['neutral']} neutral "
            f"(score {counts['score']:+.2f})"
        )
    lines.append(
        f"\n**Overall**: {overall['positive']} positive, "
        f"{overall['negative']} negative, {overall['neutral']} neutral "
        f"(score {overall['score']:+.2f})"
    )

    # Headline breakdown
    lines.append("\n### Headlines by Sentiment")
    for label in ("positive", "negative", "neutral"):
        matching = [r["title"] for r in results if r.get("label") == label and r.get("title")]
        if matching:
            lines.append(f"\n**{label.capitalize()}**")
            for title in matching:
                lines.append(f"  - {title}")

    return {
        "overall":   overall,
        "by_ticker": dict(by_ticker),
        "summary":   "\n".join(lines),
    }


# ── LangGraph node ─────────────────────────────────────────────────────────────

def sentiment_node(state: WealthManagerState) -> dict:
    """
    Sentiment agent node.
    Expects state["tickers"] to be populated by the orchestrator.
    Runs tool 1 (fetch news) then tool 2 (inference) and aggregates results.
    """
    print("--- 🔍 AGENT: SENTIMENT ANALYSIS ---")

    tickers = state.get("tickers") or []
    if not tickers:
        print("  ⚠️  No tickers in state — skipping sentiment analysis.")
        return {
            "sentiment_score": 0.0,
            "sentiment_results": [],
            "sentiment_summary": {},
            "messages": ["Sentiment: no tickers provided."],
        }

    print(f"  Tickers: {tickers}")

    # Tool 1 — fetch news
    articles = fetch_ticker_news.invoke({"tickers": tickers})   # type: ignore[attr-defined]
    print(f"  Fetched {len(articles)} articles")

    # Tool 2 — run inference
    results = run_sentiment_inference.invoke({"articles": articles})  # type: ignore[attr-defined]
    print(f"  Inference complete on {len(results)} articles")

    # Aggregate
    agg = _aggregate(results)
    score = agg["overall"]["score"]

    print(f"  Overall score: {score:+.4f}")

    return {
        "news_articles":     articles,
        "sentiment_results": results,
        "sentiment_summary": agg,
        "sentiment_score":   score,
        "messages":          [agg["summary"]],
    }
