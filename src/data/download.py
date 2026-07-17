"""
Pulls a real review dataset from the Hugging Face Hub and normalizes it into
the same schema the rest of the pipeline expects (review_id, rating, text).

Needs network access to huggingface.co - if you're working somewhere without
that, use src/data/synthetic.py instead, the schema matches.
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


SOURCES = {
    # binary polarity, 1-5 star rating not always present -> we map label to a
    # pseudo rating (0 -> 1, 1 -> 5) since we mainly need text + some rating.
    "amazon_polarity": {
        "hf_name": "amazon_polarity",
        "text_col": "content",
        "label_col": "label",
    },
    # yelp has stars 1-5 directly, closer to the real e-commerce use case
    "yelp_review_full": {
        "hf_name": "yelp_review_full",
        "text_col": "text",
        "label_col": "label",  # 0-4, we +1 it
    },
}


def _label_to_rating(source: str, label: int) -> int:
    if source == "amazon_polarity":
        return 1 if label == 0 else 5
    if source == "yelp_review_full":
        return label + 1
    raise ValueError(f"unknown source {source}")


def main():
    parser = argparse.ArgumentParser(description="Download + normalize a review dataset")
    parser.add_argument("--source", choices=SOURCES.keys(), default="amazon_polarity")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--out", default="data/processed/reviews.csv")
    args = parser.parse_args()

    cfg = SOURCES[args.source]
    print(f"loading {cfg['hf_name']} [{args.split}] from the Hub...")
    ds = load_dataset(cfg["hf_name"], split=f"{args.split}[:{args.limit}]")

    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "review_id": i + 1,
            "product_category": "unknown",
            "rating": _label_to_rating(args.source, ex[cfg["label_col"]]),
            "text": ex[cfg["text_col"]].strip().replace("\n", " "),
            "true_sentiment": None,   # not labeled independently of rating in these sets
            "is_mismatch": None,
            "is_templated_fake": None,
        })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
