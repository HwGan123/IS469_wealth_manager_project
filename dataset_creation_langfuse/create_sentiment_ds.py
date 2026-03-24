"""
dataset_creation/create_sentiment_ds.py

Creates a fixed Langfuse evaluation dataset from the SaguaroCapital
Gold Commodity Sentiment HuggingFace dataset.

Run once (idempotent — stable item ids prevent duplicates):
    python dataset_creation/create_sentiment_ds.py

Required env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
"""

from datasets import load_dataset
from langfuse import Langfuse
import pandas as pd

DATASET_NAME = "gold-commodity-sentiment-eval"
HF_DATASET = "SaguaroCapital/sentiment-analysis-in-commodity-market-gold"


def build_gold_df(gold_ds) -> pd.DataFrame:
    """
    Normalise the Gold dataset into a DataFrame with columns:
      'sentence'   — news text (from 'News' column)
      'label_text' — 'positive' / 'negative' / 'neutral' (rows with none dropped)
    """
    for split in ["test", "validation", "train"]:
        if split in gold_ds:
            ds = gold_ds[split]
            break
    else:
        ds = gold_ds[list(gold_ds.keys())[0]]

    rows = ds.to_pandas()

    rows["label_text"] = rows["Price Sentiment"].astype(str).str.strip().str.lower()
    rows["sentence"] = rows["News"].astype(str).str.strip()

    before = len(rows)
    rows = rows[
        rows["label_text"].isin({"positive", "negative", "neutral"})
    ].reset_index(drop=True)
    print(f"Dropped {before - len(rows)} rows with missing/none labels")

    result = rows[["sentence", "label_text"]]
    print(f"Gold dataset size: {len(result)}")
    print("Label distribution:")
    print(result["label_text"].value_counts())
    return result


def main():
    print(f"⏩ Loading '{HF_DATASET}' from HuggingFace ...")
    gold_ds = load_dataset(HF_DATASET)
    print("✅ Dataset loaded.")

    df = build_gold_df(gold_ds)

    langfuse = Langfuse()

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="SaguaroCapital Gold Commodity Sentiment — fixed cross-model eval set",
        metadata={"source": HF_DATASET},
    )

    for idx, row in df.iterrows():
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"gold-{idx}",  # stable id — safe to re-run
            input={"sentence": row["sentence"]},
            expected_output={"label": row["label_text"]},
        )

    langfuse.flush()
    print(f"✅ Uploaded {len(df)} items to Langfuse dataset '{DATASET_NAME}'")


if __name__ == "__main__":
    main()
