"""Sentiment Agent — LangGraph node using fine-tuned Llama-3.2-3B-Instruct.

Pipeline:
  fetch_news_yfinance → classify_headline (per item) → aggregate_sentiment
  → generate_thematic_summary → sentiment_node (LangGraph entry-point)

Model:
  LlamaForCausalLM loaded from the local QLoRA checkpoint.
  All prompts use the Llama-3 instruct chat format and demand JSON responses.
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import yfinance as yf
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph.state import WealthManagerState

# ---------------------------------------------------------------------------
# Paths & tuneable constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_MODEL_PATH = str(
    _PROJECT_ROOT
    / "experiments"
    / "sentiment_agent"
    / "saved_models"
    / "llama_3.2_3b_instruct_saved"
)

MODEL_PATH: str = os.getenv("LLAMA_SENTIMENT_MODEL_PATH", _DEFAULT_MODEL_PATH)
MAX_NEW_TOKENS_CLASSIFY: int = int(os.getenv("SENTIMENT_MAX_NEW_TOKENS_CLASSIFY", "30"))
CLASSIFY_BATCH_SIZE: int = int(os.getenv("SENTIMENT_CLASSIFY_BATCH_SIZE", "16"))
MAX_NEW_TOKENS_THEME: int = int(os.getenv("SENTIMENT_MAX_NEW_TOKENS_THEME", "256"))
MAX_INPUT_LENGTH: int = int(os.getenv("SENTIMENT_MAX_INPUT_LENGTH", "512"))
MAX_INPUT_LENGTH_THEME: int = int(os.getenv("SENTIMENT_MAX_INPUT_LENGTH_THEME", "1024"))
SUMMARY_TRUNCATE_CHARS: int = int(os.getenv("SENTIMENT_SUMMARY_TRUNCATE_CHARS", "200"))

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """Raw news article fetched from yfinance."""

    ticker: str
    title: str
    summary: Optional[str] = None
    url: str = ""
    published_at: str = ""
    source: str = ""


class ScoredNewsItem(BaseModel):
    """A news item enriched with model sentiment classification."""

    ticker: str
    title: str
    summary: Optional[str] = None
    label: str  # "positive" | "neutral" | "negative"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    url: str = ""
    published_at: str = ""
    source: str = ""


class SentimentDistribution(BaseModel):
    """Fractional breakdown of labels across all scored items."""

    positive: float
    neutral: float
    negative: float


class SentimentThemes(BaseModel):
    """Structured themes extracted by the LLM from labelled headlines."""

    doing_well_in: list[str]
    bearish_concerns: list[str]
    market_tone_summary: str


class SentimentOutput(BaseModel):
    """Full structured output returned by run_sentiment_pipeline."""

    ticker: str
    overall_sentiment: str  # "bullish" | "neutral" | "bearish"
    strength: str  # "highly_bullish" | "moderately_bullish" | "mixed_or_neutral"
    # | "moderately_bearish" | "highly_bearish"
    cwns: float  # Confidence-Weighted Net Sentiment Score ∈ [-1, +1]
    distribution: SentimentDistribution
    themes: SentimentThemes
    broad_explanation: str
    scored_items: list[ScoredNewsItem]


# ---------------------------------------------------------------------------
# Device selection & model loading
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_bundle() -> dict[str, Any]:
    """Load the Llama causal LM once at import time so node calls are fast.

    Returns a bundle dict with keys: ready, device, tokenizer, model, error.
    When loading fails the bundle marks ready=False; inference will use fallbacks.
    """
    device = _select_device()
    bundle: dict[str, Any] = {
        "ready": False,
        "device": device,
        "tokenizer": None,
        "model": None,
        "error": None,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left-padding is correct for decoder-only models: the last real token
        # is always at position -1 in every sample, regardless of sequence length.
        tokenizer.padding_side = "left"

        # bfloat16 on CUDA; float16 on MPS (M1-safe); float32 on CPU
        if device.type == "cuda":
            dtype = torch.bfloat16
        elif device.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # sdpa = scaled dot-product attention (PyTorch 2.0+) — faster on CPU/MPS
        # low_cpu_mem_usage=True streams each shard directly into tensors without
        # allocating a second full-model buffer, cutting peak RAM and load time.
        common_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            common_kwargs["device_map"] = "auto"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                attn_implementation="sdpa",
                **common_kwargs,
            )
            print("[sentiment] Using sdpa attention.")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **common_kwargs)
        if device.type != "cuda":
            model.to(device)
        model.eval()

        # torch.compile adds significant warmup on first call — skip unless on CUDA
        # where the persistent kernel cache makes it worth the cost.
        if device.type == "cuda":
            try:
                model = torch.compile(model)
                print("[sentiment] torch.compile applied.")
            except Exception:
                pass

        bundle.update(
            {"ready": True, "tokenizer": tokenizer, "model": model, "dtype": dtype}
        )
        print(f"[sentiment] Loaded '{MODEL_PATH}' on {device} ({dtype}).")
    except Exception as exc:
        bundle["error"] = str(exc)
        print(f"[sentiment] Model load failed, using fallbacks: {exc}")
    return bundle


_MODEL_BUNDLE: dict[str, Any] | None = None
_MODEL_BUNDLE_LOCK = threading.Lock()


def _get_model_bundle() -> dict[str, Any]:
    """Return the model bundle, loading it lazily on first call (thread-safe)."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    with _MODEL_BUNDLE_LOCK:
        if _MODEL_BUNDLE is None:
            _MODEL_BUNDLE = _load_model_bundle()
    return _MODEL_BUNDLE


def _generate(
    prompt: str, max_new_tokens: int, max_input_length: int | None = None
) -> str:
    """Run greedy generation for a single prompt; used only for thematic summary.

    Returns an empty string when the model bundle is not ready.
    """
    bundle = _get_model_bundle()
    if not bundle["ready"]:
        return ""

    tokenizer: AutoTokenizer = bundle["tokenizer"]
    model: AutoModelForCausalLM = bundle["model"]
    device: torch.device = bundle["device"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length or MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len: int = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _label_token_ids() -> dict[str, int]:
    """Return the first token ID for each label as generated after '{"label": "'.

    We prime the prompt with that prefix so the very first generated token is the
    start of the label word — this lets us read real logit confidence from scores[0].
    """
    tokenizer: AutoTokenizer = _get_model_bundle()["tokenizer"]
    prefix = '{"label": "'
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = {}
    for label in ("positive", "neutral", "negative"):
        full_ids = tokenizer.encode(prefix + label, add_special_tokens=False)
        ids[label] = full_ids[len(prefix_ids)]  # first token of the label word
    return ids


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _classify_prompt(title: str, summary: Optional[str]) -> str:
    """Build a Llama-3 instruct prompt primed to produce a JSON classification.

    The prompt ends with '{"label": "' so the very first generated token is the
    label word (positive/neutral/negative). This lets us derive confidence from
    the model's logits at that position rather than a self-reported score.
    Summary is truncated to SUMMARY_TRUNCATE_CHARS to keep sequences short.
    """
    if summary:
        summary = summary[:SUMMARY_TRUNCATE_CHARS]
    text = f"{title}. {summary}" if summary else title
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a financial sentiment classifier. "
        "Classify the sentiment of the given financial news. "
        'Respond ONLY with valid JSON: {"label": "<positive|neutral|negative>", "reason": "<max 12 words>"}\n'
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"News: {text}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        '{"label": "'  # primed — first generated token IS the label
    )


def _theme_prompt(
    ticker: str,
    scored_items: list[ScoredNewsItem],
    overall_sentiment: str = "",
    logger: Optional[Callable[[str], None]] = None,
) -> str:
    """Build a Llama-3 instruct prompt for structured thematic extraction."""
    headlines_txt = "\n".join(
        f"- ({item.label}) {item.title}" for item in scored_items[:15]
    )
    sentiment_ctx = (
        f" The overall sentiment for {ticker} is {overall_sentiment}."
        if overall_sentiment
        else ""
    )
    if logger:
        logger("✅ Constructed JSON extraction prompt")
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a senior financial analyst assistant. "
        f"Analyse ONLY the news headlines for the stock {ticker} and extract structured investment themes. "
        f"Focus exclusively on {ticker} — ignore any other companies or tickers mentioned in the headlines. "
        "The 'doing_well_in' and 'bearish_concerns' fields must contain specific business topics (e.g. 'AI chip demand', 'earnings growth', 'supply chain pressure') — NOT sentiment words like 'positive' or 'negative'. "
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"doing_well_in": ["specific topic1", "specific topic2"], '
        '"bearish_concerns": ["specific concern1", "specific concern2"], '
        '"market_tone_summary": "1-2 sentence summary about ' + ticker + ' only.", '
        '"broad_explanation": "2-3 sentence explanation of WHY the sentiment for '
        + ticker
        + ' is as classified, citing specific headline themes."}\n'
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Ticker: {ticker}{sentiment_ctx}\nHeadlines:\n{headlines_txt}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _parse_json_output(raw: str) -> dict:
    """Extract the first JSON object from model output.

    Tries strict json.loads first, then a regex scan for the first {...} block.
    Returns an empty dict on total failure so callers can apply their own defaults.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def _parse_pub_date(raw: Any) -> Optional[datetime]:
    """Best-effort parse of yfinance pubDate into an aware UTC datetime."""
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        # Unix timestamp
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    s = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def fetch_news_yfinance(
    ticker: str,
    limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[NewsItem]:
    """Fetch up to *limit* recent news articles for *ticker* via yfinance.

    Args:
        ticker: Stock ticker symbol.
        limit: Maximum number of articles to return.
        start_date: Optional ISO date string "YYYY-MM-DD"; exclude articles before this.
        end_date: Optional ISO date string "YYYY-MM-DD"; exclude articles after this.

    Returns an empty list on error so the pipeline degrades gracefully.
    """
    # Parse date bounds once
    dt_start: Optional[datetime] = None
    dt_end: Optional[datetime] = None
    if start_date:
        try:
            dt_start = datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            print(f"[sentiment] Invalid start_date '{start_date}', ignoring.")
    if end_date:
        try:
            dt_end = datetime.strptime(end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            # Include the full end day
            dt_end = dt_end.replace(hour=23, minute=59, second=59)
        except ValueError:
            print(f"[sentiment] Invalid end_date '{end_date}', ignoring.")

    items: list[NewsItem] = []
    try:
        # Fetch more than limit so date filtering still yields enough results
        fetch_count = limit * 3 if (dt_start or dt_end) else limit
        raw_news = yf.Ticker(ticker).get_news(count=fetch_count) or []
        for article in raw_news:
            if len(items) >= limit:
                break
            content = article.get("content") or {}
            title = content.get("title") or article.get("title") or ""
            if not title:
                continue
            pub_raw = content.get("pubDate") or article.get("providerPublishTime") or ""
            pub_dt = _parse_pub_date(pub_raw)

            # Apply date filters
            if dt_start and pub_dt and pub_dt < dt_start:
                continue
            if dt_end and pub_dt and pub_dt > dt_end:
                continue

            summary = content.get("summary") or ""
            items.append(
                NewsItem(
                    ticker=ticker,
                    title=title,
                    summary=summary or None,
                    url=(
                        content.get("canonicalUrl", {}).get("url")
                        or article.get("link")
                        or ""
                    ),
                    published_at=str(pub_raw),
                    source=(
                        content.get("provider", {}).get("displayName") or "yfinance"
                    ),
                )
            )
    except Exception as exc:
        print(f"[sentiment] fetch_news_yfinance failed for {ticker}: {exc}")
    return items


def aggregate_sentiment(
    ticker: str, scored_items: list[ScoredNewsItem]
) -> tuple[SentimentDistribution, str, str, float]:
    """Compute label distribution, overall sentiment, strength, and CWNS.

    Industry standard: Confidence-Weighted Net Sentiment Score (CWNS), as used
    by RavenPack, Refinitiv News Analytics, and the FinBERT literature.

    Each article contributes:
        +confidence  if label == "positive"
        -confidence  if label == "negative"
         0           if label == "neutral"

    CWNS = mean(signed scores) ∈ [-1, +1]

    Thresholds (calibrated to FinBERT / RavenPack conventions):
        CWNS >  0.25 → highly_bullish
        CWNS >  0.10 → moderately_bullish
        CWNS > -0.10 → mixed_or_neutral
        CWNS > -0.25 → moderately_bearish
        CWNS ≤ -0.25 → highly_bearish

    Returns:
        (SentimentDistribution, overall_sentiment, strength, cwns)
    """
    if not scored_items:
        return (
            SentimentDistribution(positive=0.0, neutral=1.0, negative=0.0),
            "neutral",
            "mixed_or_neutral",
            0.0,
        )

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    signed_scores: list[float] = []
    for item in scored_items:
        key = item.label if item.label in counts else "neutral"
        counts[key] += 1
        if item.label == "positive":
            signed_scores.append(item.confidence)
        elif item.label == "negative":
            signed_scores.append(-item.confidence)
        else:
            signed_scores.append(0.0)

    total = len(scored_items)
    dist = SentimentDistribution(
        positive=round(counts["positive"] / total, 4),
        neutral=round(counts["neutral"] / total, 4),
        negative=round(counts["negative"] / total, 4),
    )

    cwns = round(sum(signed_scores) / total, 4)

    if cwns > 0.25:
        overall, strength = "bullish", "highly_bullish"
    elif cwns > 0.10:
        overall, strength = "bullish", "moderately_bullish"
    elif cwns >= -0.10:
        overall, strength = "neutral", "mixed_or_neutral"
    elif cwns >= -0.25:
        overall, strength = "bearish", "moderately_bearish"
    else:
        overall, strength = "bearish", "highly_bearish"

    return dist, overall, strength, cwns


def generate_thematic_summary(
    ticker: str,
    distribution: SentimentDistribution,
    scored_items: list[ScoredNewsItem],
    overall_sentiment: str = "",
    logger: Optional[Callable[[str], None]] = None,
) -> tuple[SentimentThemes, str]:
    """Use the Llama model to extract structured themes from labelled headlines.

    Returns:
        (SentimentThemes, broad_explanation)
    Provides sensible defaults when the model is unavailable or output unparseable.
    """
    prompt = _theme_prompt(
        ticker,
        scored_items,
        overall_sentiment,
        logger=logger,
    )
    raw = _generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS_THEME,
        max_input_length=MAX_INPUT_LENGTH_THEME,
    )
    parsed = _parse_json_output(raw)

    dominant = _dominant_label(distribution)

    doing_well = parsed.get("doing_well_in", [])
    if not isinstance(doing_well, list) or not doing_well:
        doing_well = ["Positive analyst coverage", "Revenue momentum"]

    bearish = parsed.get("bearish_concerns", [])
    if not isinstance(bearish, list) or not bearish:
        bearish = ["General market uncertainty"]

    tone_summary = str(
        parsed.get(
            "market_tone_summary",
            f"Market tone for {ticker} is broadly {dominant}.",
        )
    )
    broad_explanation = str(
        parsed.get(
            "broad_explanation",
            f"Recent coverage for {ticker} reflects a {dominant} sentiment overall.",
        )
    )

    themes = SentimentThemes(
        doing_well_in=[str(t) for t in doing_well[:4]],
        bearish_concerns=[str(c) for c in bearish[:4]],
        market_tone_summary=tone_summary,
    )
    return themes, broad_explanation


def _dominant_label(dist: SentimentDistribution) -> str:
    """Return the label with the highest fraction."""
    return max(
        [
            ("positive", dist.positive),
            ("neutral", dist.neutral),
            ("negative", dist.negative),
        ],
        key=lambda x: x[1],
    )[0]


# ---------------------------------------------------------------------------
# Full pipeline entry-point
# ---------------------------------------------------------------------------


def _classify_batch(
    news_items: list[NewsItem],
    logger: Optional[Callable[[str], None]] = None,
) -> list[ScoredNewsItem]:
    """Classify headlines using a single forward pass per mini-batch (no generation).

    The classify prompt is primed with '{"label": "' so the label token is the
    very next token after the prompt. We run one forward pass, read the logits at
    the last input position, and softmax over the three label token IDs.
    This avoids autoregressive generation entirely — ~10-30x faster than generate().
    """
    label_order = ["positive", "neutral", "negative"]
    scored: list[ScoredNewsItem] = []

    bundle = _get_model_bundle()
    if not bundle["ready"]:
        for item in news_items:
            scored.append(
                ScoredNewsItem(
                    ticker=item.ticker,
                    title=item.title,
                    summary=item.summary,
                    label="neutral",
                    confidence=0.5,
                    reason="Model unavailable.",
                    url=item.url,
                    published_at=item.published_at,
                    source=item.source,
                )
            )
        return scored

    tokenizer: AutoTokenizer = bundle["tokenizer"]
    model: AutoModelForCausalLM = bundle["model"]
    device: torch.device = bundle["device"]
    tok_ids = _label_token_ids()
    label_tok_indices = [tok_ids[lbl] for lbl in label_order]

    total_input_tokens = 0

    for chunk_start in range(0, len(news_items), CLASSIFY_BATCH_SIZE):
        chunk = news_items[chunk_start : chunk_start + CLASSIFY_BATCH_SIZE]
        prompts = [_classify_prompt(item.title, item.summary) for item in chunk]

        t0 = time.perf_counter()
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        batch_tokens = int(inputs["input_ids"].numel())
        total_input_tokens += batch_tokens

        with torch.inference_mode():
            # Single forward pass — no token generation loop, no KV cache needed
            logits = model(**inputs, use_cache=False).logits  # (batch, seq_len, vocab)

        elapsed = time.perf_counter() - t0

        # With left-padding the last real token is always at position -1 for every
        # sample in the batch — no per-sample indexing required.
        label_logits = logits[:, -1, label_tok_indices]  # (batch, 3)
        label_probs = torch.softmax(label_logits.float(), dim=-1)  # (batch, 3)

        for i, item in enumerate(chunk):
            probs_i = label_probs[i]
            label_idx = int(probs_i.argmax().item())
            label = label_order[label_idx]
            confidence = round(float(probs_i[label_idx].item()), 4)
            scored.append(
                ScoredNewsItem(
                    ticker=item.ticker,
                    title=item.title,
                    summary=item.summary,
                    label=label,
                    confidence=confidence,
                    reason="Classified via logit probe.",
                    url=item.url,
                    published_at=item.published_at,
                    source=item.source,
                )
            )

        done = min(chunk_start + CLASSIFY_BATCH_SIZE, len(news_items))
        seq_len = inputs["input_ids"].shape[1]
        if logger:
            logger(
                f"  [classify] batch {chunk_start // CLASSIFY_BATCH_SIZE + 1} | "
                f"{len(chunk)} articles | "
                f"seq_len={seq_len} tokens | "
                f"total_input_tokens={batch_tokens} | "
                f"latency={elapsed:.3f}s | "
                f"{elapsed / len(chunk):.3f}s/article | "
                f"progress={done}/{len(news_items)}"
            )

    if logger:
        logger(f"  [classify] total input tokens consumed: {total_input_tokens}")
    return scored


def run_sentiment_pipeline(
    ticker: str,
    news_limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
    prefetched_items: Optional[list] = None,
) -> SentimentOutput:
    """End-to-end sentiment pipeline for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        news_limit: Maximum number of articles to classify (default 20).
        start_date: Optional "YYYY-MM-DD" — only include articles on/after this date.
        end_date: Optional "YYYY-MM-DD" — only include articles on/before this date.
        prefetched_items: Optional list of NewsItem (or compatible dicts) already
            fetched by market_context_agent.  When provided, the yfinance fetch is
            skipped entirely.

    Steps:
        1. Fetch recent news via yfinance — OR use prefetched_items if supplied.
        2. Classify each headline with the Llama model (mini-batched).
        3. Aggregate into distribution + high-level labels.
        4. Generate thematic explanation with generate_thematic_summary.

    Returns a fully populated SentimentOutput Pydantic model.
    """
    ticker = ticker.upper().strip()
    pipeline_t0 = time.perf_counter()

    date_ctx = ""
    if start_date or end_date:
        date_ctx = f" (date range: {start_date or 'any'} → {end_date or 'now'})"

    log = logger or (lambda _msg: None)

    fetch_elapsed = 0.0
    if prefetched_items:
        # Convert dicts to NewsItem if needed
        news_items: list[NewsItem] = []
        for a in prefetched_items:
            if isinstance(a, NewsItem):
                news_items.append(a)
            elif isinstance(a, dict) and a.get("title"):
                news_items.append(
                    NewsItem(
                        ticker=a.get("ticker", ticker),
                        title=a["title"],
                        summary=a.get("description") or a.get("summary") or None,
                        url=a.get("url", ""),
                        published_at=str(
                            a.get("published_at") or a.get("pubDate") or ""
                        ),
                        source=a.get("source", "market_context"),
                    )
                )
        log(
            f"✅ Using {len(news_items)} pre-fetched articles from market_context_agent for {ticker}"
        )
    else:
        log(f"⏳ Fetching news for {ticker}{date_ctx}...")
        fetch_t0 = time.perf_counter()
        news_items = fetch_news_yfinance(
            ticker, limit=news_limit, start_date=start_date, end_date=end_date
        )
        fetch_elapsed = time.perf_counter() - fetch_t0

    if not news_items:
        log(f"⚠️  No news found for {ticker}{date_ctx}. Returning neutral defaults.")
        return _empty_output(ticker)
    if not prefetched_items:
        log(
            f"✅ Fetched {len(news_items)} articles for {ticker} ({fetch_elapsed:.2f}s)"
        )

    log(
        f"⏳ Classifying {len(news_items)} headlines (batch_size={CLASSIFY_BATCH_SIZE})..."
    )
    classify_t0 = time.perf_counter()
    scored_items = _classify_batch(news_items, logger=log)
    classify_elapsed = time.perf_counter() - classify_t0

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in scored_items:
        counts[s.label] += 1
    log(
        f"✅ Classified {len(scored_items)} articles in {classify_elapsed:.2f}s "
        f"({classify_elapsed / len(scored_items):.3f}s/article) — "
        f"positive: {counts['positive']}, "
        f"neutral: {counts['neutral']}, "
        f"negative: {counts['negative']}"
    )

    distribution, overall_sentiment, strength, cwns = aggregate_sentiment(
        ticker, scored_items
    )

    log(f"⏳ Generating thematic summary for {ticker}...")
    theme_t0 = time.perf_counter()
    themes, broad_explanation = generate_thematic_summary(
        ticker,
        distribution,
        scored_items,
        overall_sentiment,
        logger=log,
    )
    theme_elapsed = time.perf_counter() - theme_t0
    pipeline_elapsed = time.perf_counter() - pipeline_t0
    log(
        f"✅ Sentiment pipeline complete — {ticker} is {overall_sentiment} ({strength})\n"
        f"   Latency breakdown: fetch={fetch_elapsed:.2f}s | "
        f"classify={classify_elapsed:.2f}s | "
        f"theme_gen={theme_elapsed:.2f}s | "
        f"total={pipeline_elapsed:.2f}s"
    )

    return SentimentOutput(
        ticker=ticker,
        overall_sentiment=overall_sentiment,
        strength=strength,
        cwns=cwns,
        distribution=distribution,
        themes=themes,
        broad_explanation=broad_explanation,
        scored_items=scored_items,
    )


def _empty_output(ticker: str) -> SentimentOutput:
    return SentimentOutput(
        ticker=ticker,
        overall_sentiment="neutral",
        strength="mixed_or_neutral",
        cwns=0.0,
        distribution=SentimentDistribution(positive=0.0, neutral=1.0, negative=0.0),
        themes=SentimentThemes(
            doing_well_in=[],
            bearish_concerns=[],
            market_tone_summary="Insufficient news data to determine sentiment.",
        ),
        broad_explanation="No recent news articles were available for this ticker.",
        scored_items=[],
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def sentiment_node(state: WealthManagerState) -> dict:
    """LangGraph node: run full sentiment pipeline for all tickers in state.

    Reads:
        state["tickers"] — runs the pipeline for every ticker and aggregates.

    Returns a partial state update dict:
        sentiment_result    — primary ticker's SentimentOutput as dict
        sentiment_summary   — same dict (backward-compatible alias)
        sentiment_score     — portfolio-wide CWNS ∈ [-1, +1] (avg across tickers)
        news_articles       — combined list[dict] of ScoredNewsItem for downstream
        sentiment_results   — same list (backward-compatible alias)
        messages            — one status line with per-ticker breakdown
    """
    print("--- AGENT: SENTIMENT ANALYSIS (Llama-3.2-3B-Instruct) ---")

    tickers = state.get("tickers") or []
    if not tickers:
        return {
            "sentiment_result": {},
            "sentiment_summary": {},
            "sentiment_score": 0.0,
            "news_articles": [],
            "sentiment_results": [],
            "messages": ["Sentiment: no tickers provided."],
        }

    start_date: Optional[str] = state.get("sentiment_start_date") or None
    end_date: Optional[str] = state.get("sentiment_end_date") or None
    sentiment_logs: list[str] = []
    log_queue = state.get("__sse_log_queue")
    # The loop reference lets us safely enqueue from this thread into the
    # asyncio event loop (asyncio.Queue is not thread-safe for put_nowait).
    _sse_loop = state.get("__sse_loop")

    def _log(message: str) -> None:
        sentiment_logs.append(message)
        print(message)
        if log_queue is not None:
            try:
                item = {"node": "sentiment_agent", "message": message}
                if _sse_loop is not None and _sse_loop.is_running():
                    _sse_loop.call_soon_threadsafe(log_queue.put_nowait, item)
                else:
                    log_queue.put_nowait(item)
            except Exception:
                pass

    # Reuse news already fetched by market_context_agent to avoid duplicate calls
    prefetched: Optional[list] = None
    market_ctx = state.get("market_context") or {}
    fetch_news_result = market_ctx.get("fetch_news") or {}
    if fetch_news_result.get("articles"):
        prefetched = fetch_news_result["articles"]
        _log(
            f"  [sentiment] Reusing {len(prefetched)} articles from market_context_agent"
        )

    # Run pipeline for every ticker and collect results
    all_results: list[SentimentOutput] = []
    for ticker in tickers:
        _log(f"  [sentiment] Running pipeline for {ticker} ...")
        try:
            res: SentimentOutput = run_sentiment_pipeline(
                ticker,
                start_date=start_date,
                end_date=end_date,
                logger=_log,
                prefetched_items=prefetched,
            )
        except Exception as exc:
            err_msg = f"Sentiment pipeline failed for {ticker}: {exc}"
            _log(f"❌ {err_msg}")
            sentiment_logs.append(err_msg)
            res = _empty_output(ticker)
        all_results.append(res)

    # Aggregate across all tickers: pool every scored article, then recompute
    # distribution and CWNS on the full pool for a portfolio-wide signal.
    primary_ticker = tickers[0]
    all_scored: list = []
    for r in all_results:
        all_scored.extend(r.scored_items)

    _, overall_sentiment, strength, avg_cwns = aggregate_sentiment(
        primary_ticker, all_scored
    )

    # Use the primary ticker's detailed result for backward-compatible fields
    # (themes, broad_explanation) that downstream agents may inspect.
    primary_result = all_results[0]
    result_dict = primary_result.model_dump()

    scored_dicts = [item.model_dump() for item in all_scored]

    article_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for item in all_scored:
        article_counts[item.label] += 1

    per_ticker_summary = ", ".join(
        f"{r.ticker} {r.overall_sentiment} ({r.cwns:+.3f})" for r in all_results
    )

    return {
        "sentiment_result": result_dict,
        "sentiment_summary": result_dict,
        "sentiment_logs": sentiment_logs,
        "sentiment_score": avg_cwns,
        "news_articles": scored_dicts,
        "sentiment_results": scored_dicts,
        "messages": [
            f"Sentiment for {', '.join(tickers)}: {overall_sentiment} ({strength}), "
            f"portfolio CWNS={avg_cwns:+.3f}. "
            f"Analysed {len(all_scored)} total articles across {len(tickers)} ticker(s) — "
            f"{article_counts['positive']} positive, "
            f"{article_counts['neutral']} neutral, "
            f"{article_counts['negative']} negative. "
            f"Per-ticker: [{per_ticker_summary}]."
        ],
    }


# ---------------------------------------------------------------------------
# Local smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ticker = "NVDA"
    print(f"\n=== Running sentiment pipeline for {_ticker} ===\n")
    output = run_sentiment_pipeline(_ticker)
    print(json.dumps(output.model_dump(), indent=2))
