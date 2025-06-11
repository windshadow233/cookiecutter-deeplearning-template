import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from metrics.metric import _Metric


class _ClassificationMetric(_Metric):
    def __init__(self):
        self.preds = []
        self.targets = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor = None, preds: torch.Tensor = None, targets: torch.Tensor = None):
        if preds is not None:
            self.preds.extend(preds.tolist())
        else:
            preds = torch.argmax(logits, dim=-1)
            self.preds.extend(preds.tolist())
        if targets is not None:
            self.targets.extend(targets.tolist())

    def compute(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def reset(self):
        self.preds.clear()
        self.targets.clear()


class F1Score(_ClassificationMetric):
    def __init__(self, average='macro'):
        super().__init__()
        self.average = average

    def compute(self):
        if not self.preds or not self.targets:
            return 0.0
        return f1_score(self.targets, self.preds, average=self.average)


class Precision(_ClassificationMetric):
    def __init__(self, average='macro'):
        super().__init__()
        self.average = average

    def compute(self):
        if not self.preds or not self.targets:
            return 0.0
        return precision_score(self.targets, self.preds, average=self.average)


class Recall(_ClassificationMetric):
    def __init__(self, average='macro'):
        super().__init__()
        self.average = average

    def compute(self):
        if not self.preds or not self.targets:
            return 0.0
        return recall_score(self.targets, self.preds, average=self.average)


class Accuracy(_ClassificationMetric):
    def __init__(self):
        super().__init__()

    def compute(self):
        if not self.preds or not self.targets:
            return 0.0
        return accuracy_score(self.targets, self.preds)


class AUC(_ClassificationMetric):
    def __init__(self, average='macro', multi_class='ovr'):
        super().__init__()
        self.average = average
        self.multi_class = multi_class

    def compute(self):
        if not self.preds or not self.targets:
            return 0.0
        try:
            return roc_auc_score(self.targets, self.preds, average=self.average, multi_class=self.multi_class)
        except:
            return 0.0
