"""
create_sentiment_ds.py

Loads FinancialPhraseBank (75% agreement) and uploads all items as a
Langfuse dataset for offline evaluation / experiment tracking.

Usage:
    python create_sentiment_ds.py [--dataset-name financialphrasebank] [--split 0.2]

Environment variables required (or via .env):
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST  (optional, defaults to cloud)
"""

import argparse
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------
for parent in [Path(os.getcwd()), *Path(os.getcwd()).parents]:
    env_file = parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
        print(f"Loaded .env from: {parent}")
        break

# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------
try:
    from langfuse import Langfuse
except ImportError:
    sys.exit("langfuse not installed – run: pip install langfuse")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPB_URL = (
    "https://raw.githubusercontent.com/maxwellsarpong/"
    "NLP-financial-text-processing-dataset/master/Sentences_75Agree.txt"
)
DEFAULT_DATASET_NAME = "financialphrasebank_75agree"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_fpb() -> pd.DataFrame:
    """Download FinancialPhraseBank 75%-agreement split and return a DataFrame."""
    print(f"Downloading FinancialPhraseBank from:\n  {FPB_URL}")
    resp = requests.get(FPB_URL, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        sep="@",
        header=None,
        names=["sentence", "label_text"],
        engine="python",
        encoding="latin-1",
    )
    df["sentence"] = df["sentence"].str.strip()
    df["label_text"] = df["label_text"].str.strip().str.lower()

    # Verify labels
    valid = {"negative", "neutral", "positive"}
    bad = df[~df["label_text"].isin(valid)]
    if len(bad):
        print(f"Warning: {len(bad)} rows with unexpected labels – dropping.")
        df = df[df["label_text"].isin(valid)].reset_index(drop=True)

    print(f"Loaded {len(df)} samples")
    print(df["label_text"].value_counts().to_string())
    return df


def ensure_dataset(langfuse: Langfuse, name: str, description: str) -> None:
    """Create the dataset if it does not already exist."""
    try:
        existing = langfuse.get_dataset(name)
        print(f"Dataset '{name}' already exists ({len(existing.items)} items). Will append new items.")
    except Exception:
        langfuse.create_dataset(
            name=name,
            description=description,
            metadata={"source": "FinancialPhraseBank 75% agreement"},
        )
        print(f"Created dataset: '{name}'")


def upload_items(langfuse: Langfuse, dataset_name: str, df: pd.DataFrame) -> None:
    """Upload each FPB row as a dataset item."""
    print(f"Uploading {len(df)} items to '{dataset_name}' …")
    for idx, row in df.iterrows():
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"sentence": row["sentence"]},
            expected_output={"label": row["label_text"]},
            metadata={
                "row_index": int(idx),
                "label": row["label_text"],
            },
        )
        if (idx + 1) % 200 == 0:
            print(f"  Uploaded {idx + 1}/{len(df)} …")

    print(f"Done – {len(df)} items uploaded to dataset '{dataset_name}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Upload FinancialPhraseBank to Langfuse")
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("LANGFUSE_DATASET_NAME", DEFAULT_DATASET_NAME),
        help="Name of the Langfuse dataset to create/update",
    )
    parser.add_argument(
        "--description",
        default="FinancialPhraseBank 75% annotator agreement – financial sentiment (neg/neu/pos)",
    )
    args = parser.parse_args()

    # Validate Langfuse credentials
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        if not os.getenv(key):
            sys.exit(f"Missing environment variable: {key}")

    lf = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    df = load_fpb()
    ensure_dataset(lf, args.dataset_name, args.description)
    upload_items(lf, args.dataset_name, df)

    # Flush to ensure all items are sent
    lf.flush()
    print("Langfuse dataset creation complete.")


if __name__ == "__main__":
    main()
