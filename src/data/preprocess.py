"""
Shared preprocessing so the baseline, DistilBERT, and TinyBERT training
scripts all derive labels the same way and don't quietly drift apart.
"""
import re
import pandas as pd


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rating_to_sentiment_label(rating: int) -> int:
    """
    Collapse 1-5 stars down to a 3-way sentiment label for the classifier.
    This is what the model is trained to predict FROM TEXT ALONE - the whole
    point of the mismatch scorer later is comparing this against the rating
    the user actually left, so don't derive one from the other at inference
    time.
    0 = negative, 1 = neutral, 2 = positive
    """
    rating = int(rating)
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


LABEL_NAMES = ["negative", "neutral", "positive"]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    # for training we need a target derived from rating just to have labels
    # to fine-tune against, since we don't have independently hand-labeled
    # sentiment for every row (real-world you'd want a separate annotated
    # subset for this, noted in the README limitations section)
    df["label"] = df["rating"].apply(rating_to_sentiment_label)
    return df
