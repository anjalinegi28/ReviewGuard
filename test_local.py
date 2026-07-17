import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

from src.mismatch.classifiers import HFPipelineClassifier
from src.mismatch.scorer import MismatchScorer

clf = HFPipelineClassifier("artifacts/models/distilbert-final", device=-1)
scorer = MismatchScorer(clf, flag_threshold=0.40)

result = scorer.score(rating=5, text="This product broke on day one, total waste of money")
print(result)