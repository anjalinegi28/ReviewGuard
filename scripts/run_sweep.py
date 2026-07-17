import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

from src.mismatch.classifiers import HFPipelineClassifier
from src.mismatch.scorer import MismatchScorer
import pandas as pd

clf = HFPipelineClassifier("artifacts/models/distilbert-final", device=-1)
df = pd.read_csv("data/processed/reviews.csv")

truth = df[df.is_mismatch == True].sample(50, random_state=1).to_dict("records")
clean = df[df.is_mismatch == False].sample(50, random_state=1).to_dict("records")

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    scorer = MismatchScorer(clf, flag_threshold=threshold)
    mismatch_results = scorer.score_batch(truth)
    recall = sum(r.is_flagged for r in mismatch_results) / len(mismatch_results)
    clean_results = scorer.score_batch(clean)
    fpr = sum(r.is_flagged for r in clean_results) / len(clean_results)
    print(f"threshold={threshold:.2f}  recall={recall:.2%}  false_positive_rate={fpr:.2%}")

clean_full = df[df.is_mismatch == False].to_dict("records")
scorer = MismatchScorer(clf, flag_threshold=0.35)
results = scorer.score_batch(clean_full)
fpr = sum(r.is_flagged for r in results) / len(results)
print(f"full clean-set FPR @ 0.35: {fpr:.2%}")