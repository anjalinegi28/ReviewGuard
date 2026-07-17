"""
Generates a synthetic review dataset for local development when you don't
want to (or can't) pull the real Amazon/Yelp polarity dataset from the Hub.

It's not meant to replace real data for the final model - it's here so the
rest of the pipeline (mismatch scorer, dedup, API, tests) has something
realistic to run against without a network call. It deliberately seeds in:

  - genuine 1/5-star reviews that match their text
  - "mismatch" reviews: high rating + negative text, or low rating + positive text
  - clusters of near-duplicate reviews (same template, a couple words swapped)
    to make sure the dedup step has something to catch
"""
import argparse
import random
import csv
from pathlib import Path

random.seed(13)

POSITIVE_SNIPPETS = [
    "Works exactly as described, very happy with this purchase.",
    "Great quality for the price, would buy again.",
    "Shipped fast and the packaging was solid.",
    "Exceeded my expectations, using it daily now.",
    "Simple to set up and does the job well.",
    "Sturdy build, feels like it'll last for years.",
    "Better than the more expensive one I returned.",
    "My whole family uses this now, no complaints.",
]

NEGATIVE_SNIPPETS = [
    "Broke after two days, complete waste of money.",
    "Arrived damaged and customer service ignored my emails.",
    "Stopped working within a week, do not buy this.",
    "Cheaply made, nothing like the pictures.",
    "Had to return it, was defective out of the box.",
    "Battery died on day one and never charged again.",
    "Smells like chemicals and gave me a headache.",
    "Completely different product than what was pictured.",
]

NEUTRAL_SNIPPETS = [
    "It's fine, does what it says, nothing special.",
    "Average product, matches the description.",
    "Ok for the price, wouldn't expect more.",
    "Does the job, no strong feelings either way.",
]

# Templates used to generate templated / near-duplicate fake reviews.
FAKE_TEMPLATES = [
    "Amazing {noun}!! Best purchase I ever made, 5 stars all the way, {filler}",
    "I love this {noun} so much, changed my life, would recommend to everyone {filler}",
    "This {noun} is the best thing I've bought this year, {filler} highly recommend!!",
]
FAKE_NOUNS = ["product", "item", "gadget", "thing", "purchase"]
FAKE_FILLERS = ["", "seriously", "no joke", "10/10", "amazing seller too", "fast shipping"]

PRODUCT_CATEGORIES = [
    "kitchen blender", "wireless earbuds", "yoga mat", "phone case",
    "desk lamp", "running shoes", "coffee maker", "backpack",
]


def _rating_from_sentiment(sentiment: str) -> int:
    if sentiment == "positive":
        return random.choice([4, 5])
    if sentiment == "negative":
        return random.choice([1, 2])
    return 3


def _make_genuine_review(review_id: int) -> dict:
    sentiment = random.choice(["positive", "negative", "neutral"])
    snippet_pool = {
        "positive": POSITIVE_SNIPPETS,
        "negative": NEGATIVE_SNIPPETS,
        "neutral": NEUTRAL_SNIPPETS,
    }[sentiment]
    text = random.choice(snippet_pool)
    category = random.choice(PRODUCT_CATEGORIES)
    rating = _rating_from_sentiment(sentiment)
    return {
        "review_id": review_id,
        "product_category": category,
        "rating": rating,
        "text": f"{text} ({category})",
        "true_sentiment": sentiment,
        "is_mismatch": False,
        "is_templated_fake": False,
    }


def _make_mismatch_review(review_id: int) -> dict:
    # Text says one thing, rating says the opposite.
    flip = random.choice(["high_rating_negative_text", "low_rating_positive_text"])
    category = random.choice(PRODUCT_CATEGORIES)
    if flip == "high_rating_negative_text":
        text = random.choice(NEGATIVE_SNIPPETS)
        rating = random.choice([4, 5])
        sentiment = "negative"
    else:
        text = random.choice(POSITIVE_SNIPPETS)
        rating = random.choice([1, 2])
        sentiment = "positive"
    return {
        "review_id": review_id,
        "product_category": category,
        "rating": rating,
        "text": f"{text} ({category})",
        "true_sentiment": sentiment,
        "is_mismatch": True,
        "is_templated_fake": False,
    }


def _make_templated_cluster(start_id: int, size: int) -> list:
    """A handful of near-duplicate 5-star reviews from the same template."""
    template = random.choice(FAKE_TEMPLATES)
    category = random.choice(PRODUCT_CATEGORIES)
    reviews = []
    for i in range(size):
        noun = random.choice(FAKE_NOUNS)
        filler = random.choice(FAKE_FILLERS)
        text = template.format(noun=noun, filler=filler)
        reviews.append({
            "review_id": start_id + i,
            "product_category": category,
            "rating": 5,
            "text": text,
            "true_sentiment": "positive",
            "is_mismatch": False,
            "is_templated_fake": True,
        })
    return reviews


def generate(n: int) -> list:
    rows = []
    next_id = 1

    n_mismatch = int(n * 0.12)
    n_fake_clusters = max(1, int(n * 0.08) // 4)
    n_genuine = n - n_mismatch - (n_fake_clusters * 4)

    for _ in range(n_genuine):
        rows.append(_make_genuine_review(next_id))
        next_id += 1

    for _ in range(n_mismatch):
        rows.append(_make_mismatch_review(next_id))
        next_id += 1

    for _ in range(n_fake_clusters):
        cluster = _make_templated_cluster(next_id, size=4)
        rows.extend(cluster)
        next_id += len(cluster)

    random.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ReviewGuard dataset")
    parser.add_argument("--n", type=int, default=4000, help="approximate number of rows")
    parser.add_argument("--out", type=str, default="data/processed/reviews.csv")
    args = parser.parse_args()

    rows = generate(args.n)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["review_id", "product_category", "rating", "text",
                  "true_sentiment", "is_mismatch", "is_templated_fake"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_mismatch = sum(1 for r in rows if r["is_mismatch"])
    n_fake = sum(1 for r in rows if r["is_templated_fake"])
    print(f"wrote {len(rows)} rows to {out_path}")
    print(f"  mismatches: {n_mismatch}")
    print(f"  templated fakes: {n_fake}")


if __name__ == "__main__":
    main()
