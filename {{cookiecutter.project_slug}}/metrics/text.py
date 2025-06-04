from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from typing import List
from metrics.metric import _Metric


class BLEU(_Metric):
    def __init__(self, n_gram=4, smooth=False, **kwargs):
        self.metric = BLEUScore(n_gram=n_gram, smooth=smooth, **kwargs)

    def update(self, preds: List[str], targets: List[List[str]]):
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute().item()

    def reset(self):
        self.metric.reset()


class ROUGE(_Metric):
    def __init__(self, rouge_keys=None, **kwargs):
        if rouge_keys is None:
            rouge_keys = ("rouge1", "rouge2", "rougeL")
        self.metric = ROUGEScore(rouge_keys=rouge_keys, **kwargs)

    def update(self, preds: List[str], targets: List[str]):
        self.metric.update(preds, targets)

    def compute(self):
        scores = self.metric.compute()
        return {k: v.item() for k, v in scores.items()}

    def reset(self):
        self.metric.reset()
