import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from metrics.metric import _Metric


class _RegressionMetric(_Metric):
    def __init__(self):
        self.preds = []
        self.targets = []

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def reset(self):
        self.preds.clear()
        self.targets.clear()


class MSE(_RegressionMetric):
    def __init__(self, multioutput='uniform_average'):
        super().__init__()
        self.multioutput = multioutput

    def compute(self):
        return mean_squared_error(self.targets, self.preds, multioutput=self.multioutput)


class MAE(_RegressionMetric):
    def __init__(self, multioutput='uniform_average'):
        super().__init__()
        self.multioutput = multioutput

    def compute(self):
        return mean_absolute_error(self.targets, self.preds, multioutput=self.multioutput)


class R2Score(_RegressionMetric):
    def __init__(self, multioutput='uniform_average'):
        super().__init__()
        self.multioutput = multioutput

    def compute(self):
        return r2_score(self.targets, self.preds, multioutput=self.multioutput)
