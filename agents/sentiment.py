"""Sentiment agent tools + LangGraph node.

This module provides four tools used by the sentiment node:
1) `fetchnewsperticker`
2) `model_inference`
3) `generate_explanation`
4) `aggregate_sentiment_scores_for_tickers`

Performance design:
- Model is preloaded at import time.
- Inference is batched (not one headline at a time).
- Inference runs under `torch.inference_mode()`.
"""

import os
from dotenv import load_dotenv
from collections import defaultdict
from typing import Any

# Load environment variables
load_dotenv()

import requests
import torch
import yfinance as yf
from langchain_core.tools import tool
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from graph.state import WealthManagerState

LABELS = ["negative", "neutral", "positive"]
MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")
INFER_BATCH_SIZE = int(os.getenv("SENTIMENT_INFER_BATCH_SIZE", "32"))
MAX_LENGTH = int(os.getenv("SENTIMENT_MAX_LENGTH", "256"))
QUANTIZATION = os.getenv("SENTIMENT_QUANTIZATION", "none").strip().lower()


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_bundle() -> dict[str, Any]:
    """Load sentiment model at startup so node calls are fast."""
    device = _select_device()
    bundle: dict[str, Any] = {
        "ready": False,
        "device": device,
        "model_name": MODEL_NAME,
        "tokenizer": None,
        "model": None,
        "id2label": {},
        "error": None,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        # Quantization is enabled only on CUDA to avoid backend incompatibilities.
        quant_mode = QUANTIZATION if device.type == "cuda" else "none"
        model_load_kwargs: dict[str, Any] = {}
        if quant_mode in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig

                if quant_mode == "4bit":
                    model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                else:
                    model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                model_load_kwargs["device_map"] = "auto"
            except Exception as quant_exc:  # pragma: no cover
                print(
                    f"[sentiment] Quantization '{quant_mode}' unavailable, falling back to full precision: {quant_exc}"
                )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, **model_load_kwargs
        )
        if quant_mode not in {"4bit", "8bit"}:
            model.to(device)
        model.eval()
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        bundle.update(
            {
                "ready": True,
                "tokenizer": tokenizer,
                "model": model,
                "id2label": id2label,
                "quantization": quant_mode,
            }
        )
        print(
            f"[sentiment] Loaded model '{MODEL_NAME}' on {device} (quantization={quant_mode})."
        )
    except Exception as exc:  # pragma: no cover
        bundle["error"] = str(exc)
        print(f"[sentiment] Model preload failed, using fallback scoring: {exc}")
    return bundle


_MODEL_BUNDLE = _load_model_bundle()


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    fixed = {k: float(scores.get(k, 0.0)) for k in LABELS}
    total = sum(fixed.values()) or 1.0
    return {k: v / total for k, v in fixed.items()}


def _fallback_scores(text: str) -> dict[str, float]:
    # Lightweight neutral fallback if model is unavailable.
    _ = text
    return {"negative": 0.2, "neutral": 0.6, "positive": 0.2}


def _infer_single(text: str) -> dict[str, float]:
    if not _MODEL_BUNDLE["ready"]:
        return _fallback_scores(text)

    tokenizer = _MODEL_BUNDLE["tokenizer"]
    model = _MODEL_BUNDLE["model"]
    device = _MODEL_BUNDLE["device"]
    id2label = _MODEL_BUNDLE["id2label"]

    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        probs = (
            torch.softmax(model(**encoded).logits, dim=-1)[0].detach().cpu().tolist()
        )

    mapped = {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    if "positive" in mapped and "negative" in mapped and "neutral" in mapped:
        return _normalize_scores(mapped)

    if len(probs) == 3:
        # Common FinBERT mapping: 0=positive, 1=negative, 2=neutral
        return _normalize_scores(
            {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2]),
            }
        )

    return _fallback_scores(text)


def _map_probs_to_scores(
    probs: list[float], id2label: dict[int, str]
) -> dict[str, float]:
    """Map model class probabilities into negative/neutral/positive dict."""
    mapped = {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    if "positive" in mapped and "negative" in mapped and "neutral" in mapped:
        return _normalize_scores(mapped)
    if len(probs) == 3:
        # Common FinBERT mapping: 0=positive, 1=negative, 2=neutral
        return _normalize_scores(
            {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2]),
            }
        )
    return _fallback_scores("")


def _infer_batch(
    texts: list[str], batch_size: int = INFER_BATCH_SIZE
) -> list[dict[str, float]]:
    """Run fast batched inference for a list of texts."""
    if not texts:
        return []
    if not _MODEL_BUNDLE["ready"]:
        return [_fallback_scores(t) for t in texts]

    tokenizer = _MODEL_BUNDLE["tokenizer"]
    model = _MODEL_BUNDLE["model"]
    device = _MODEL_BUNDLE["device"]
    id2label = _MODEL_BUNDLE["id2label"]

    all_scores: list[dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            probs_batch = (
                torch.softmax(model(**encoded).logits, dim=-1).detach().cpu().tolist()
            )

        for probs in probs_batch:
            all_scores.append(_map_probs_to_scores(probs, id2label))

    return all_scores


def _newsapi_fetch(ticker: str, max_per_ticker: int) -> list[dict]:
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        return []

    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": ticker,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_per_ticker,
            "apiKey": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()
    items = []
    for article in resp.json().get("articles", []):
        items.append(
            {
                "ticker": ticker,
                "title": article.get("title") or "",
                "description": article.get("description") or "",
                "url": article.get("url") or "",
                "published_at": article.get("publishedAt") or "",
                "source": (article.get("source") or {}).get("name") or "newsapi",
            }
        )
    return items


def _yfinance_fetch(ticker: str, max_per_ticker: int) -> list[dict]:
    items = []
    news_list = (yf.Ticker(ticker).news or [])[:max_per_ticker]
    for article in news_list:
        content = article.get("content") or {}
        title = content.get("title") or article.get("title") or ""
        summary = content.get("summary") or ""
        items.append(
            {
                "ticker": ticker,
                "title": title,
                "description": summary,
                "url": content.get("canonicalUrl", {}).get("url")
                or article.get("link")
                or "",
                "published_at": str(
                    content.get("pubDate") or article.get("providerPublishTime") or ""
                ),
                "source": content.get("provider", {}).get("displayName") or "yfinance",
            }
        )
    return items


@tool
def fetchnewsperticker(
    tickers: list[str], provider: str = "newsapi", max_per_ticker: int = 5
) -> list[dict]:
    """Fetch per-ticker news from NewsAPI or yfinance.

    If NewsAPI is selected but not available/misconfigured, a yfinance fallback
    is used for resilience.
    """
    provider = provider.lower().strip()
    fetched: list[dict] = []

    for ticker in tickers:
        try:
            if provider == "newsapi":
                rows = _newsapi_fetch(ticker, max_per_ticker)
                if not rows:
                    rows = _yfinance_fetch(ticker, max_per_ticker)
            elif provider == "yfinance":
                rows = _yfinance_fetch(ticker, max_per_ticker)
            else:
                raise ValueError("provider must be 'newsapi' or 'yfinance'")
            fetched.extend(rows)
        except Exception as exc:
            fetched.append(
                {
                    "ticker": ticker,
                    "title": f"[ERROR fetching news: {exc}]",
                    "description": "",
                    "url": "",
                    "published_at": "",
                    "source": provider,
                }
            )

    return fetched


@tool
def model_inference(articles: list[dict]) -> list[dict]:
    """Run batched sentiment inference for each article.

    This avoids per-row tokenizer/model overhead and is significantly faster
    when many headlines are processed in one graph step.
    """
    prepared_texts: list[str] = []
    prepared_indices: list[int] = []
    results: list[dict] = [
        {**article, "scores": {}, "label": "unknown"} for article in articles
    ]

    for idx, article in enumerate(articles):
        text = " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("description") or ""),
            ]
        ).strip()
        if not text or text.startswith("[ERROR"):
            continue
        prepared_indices.append(idx)
        prepared_texts.append(text)

    batch_scores = _infer_batch(prepared_texts, batch_size=INFER_BATCH_SIZE)
    for idx, scores in zip(prepared_indices, batch_scores):
        label = max(scores, key=scores.get)
        results[idx] = {**articles[idx], "scores": scores, "label": label}

    return results


@tool
def aggregate_sentiment_scores_for_tickers(sentiment_results: list[dict]) -> dict:
    """Aggregate sentiment into individual ticker scores + overall score."""
    by_ticker: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "negative": 0.0,
            "neutral": 0.0,
            "positive": 0.0,
            "score": 0.0,
            "n": 0.0,
        }
    )
    overall = {"negative": 0.0, "neutral": 0.0, "positive": 0.0, "score": 0.0, "n": 0.0}

    for row in sentiment_results:
        ticker = str(row.get("ticker") or "UNKNOWN")
        label = str(row.get("label") or "neutral")
        if label not in LABELS:
            continue
        by_ticker[ticker][label] += 1.0
        by_ticker[ticker]["n"] += 1.0
        overall[label] += 1.0
        overall["n"] += 1.0

    def _score(bucket: dict[str, float]) -> float:
        total = bucket.get("n", 0.0) or 1.0
        return round((bucket["positive"] - bucket["negative"]) / total, 4)

    for ticker in by_ticker:
        by_ticker[ticker]["score"] = _score(by_ticker[ticker])
    overall["score"] = _score(overall)

    return {
        "overall": overall,
        "by_ticker": dict(by_ticker),
    }


@tool
def generate_explanation(sentiment_results: list[dict], aggregated: dict) -> str:
    """Generate concise explanation for per-ticker and overall sentiment."""
    by_ticker = aggregated.get("by_ticker", {})
    overall = aggregated.get("overall", {})

    lines = ["## Sentiment Explanation"]
    lines.append(f"Model: {MODEL_NAME}")
    lines.append(f"Articles analyzed: {len(sentiment_results)}")

    for ticker, stats in by_ticker.items():
        lines.append(
            f"- {ticker}: +{int(stats['positive'])} / -{int(stats['negative'])} / ={int(stats['neutral'])} "
            f"-> score {stats['score']:+.2f}"
        )

    lines.append(
        f"Overall: +{int(overall.get('positive', 0))} / -{int(overall.get('negative', 0))} / "
        f"={int(overall.get('neutral', 0))} -> score {float(overall.get('score', 0.0)):+.2f}"
    )
    return "\n".join(lines)


def sentiment_node(state: WealthManagerState) -> dict:
    """LangGraph node wrapper for the sentiment toolchain.

    Flow:
    1) fetch news
    2) run batched model inference
    3) aggregate per-ticker + overall scores
    4) generate human-readable explanation
    """
    print("--- AGENT: SENTIMENT ANALYSIS ---")
    tickers = state.get("tickers") or []
    if not tickers:
        return {
            "sentiment_score": 0.0,
            "sentiment_results": [],
            "sentiment_summary": {},
            "messages": ["Sentiment: no tickers provided."],
            "audit_iteration_count": state.get("audit_iteration_count", 0)
        }

    provider = os.getenv("SENTIMENT_NEWS_PROVIDER", "newsapi")

    articles = fetchnewsperticker.invoke(
        {"tickers": tickers, "provider": provider, "max_per_ticker": 5}
    )
    inferred = model_inference.invoke({"articles": articles})
    aggregated = aggregate_sentiment_scores_for_tickers.invoke(
        {"sentiment_results": inferred}
    )
    explanation = generate_explanation.invoke(
        {"sentiment_results": inferred, "aggregated": aggregated}
    )

    return {
        "news_articles": articles,
        "sentiment_results": inferred,
        "sentiment_summary": {
            **aggregated,
            "summary": explanation,
            "provider": provider,
            "model_name": MODEL_NAME,
        },
        "sentiment_score": float(aggregated.get("overall", {}).get("score", 0.0)),
        "messages": [explanation],
        "audit_iteration_count": state.get("audit_iteration_count", 0)
    }
