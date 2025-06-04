import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from metrics.metric import _Metric


class _SegmentationMetric(_Metric):
    def __init__(self, num_classes: int, ignore_index: int = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor = None, targets: torch.Tensor = None):
        preds = preds.flatten()
        targets = targets.flatten()
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]
        self.confusion += confusion_matrix(targets, preds, labels=list(range(self.num_classes)))

    def compute(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _safe_divide(self, numerator, denominator):
        return numerator / (denominator + 1e-10)


class PixelAccuracy(_SegmentationMetric):
    def compute(self):
        correct = np.trace(self.confusion)
        total = self.confusion.sum()
        return self._safe_divide(correct, total)


class MeanIoU(_SegmentationMetric):
    def compute(self):
        intersection = np.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        iou = self._safe_divide(intersection, union)
        return np.nanmean(iou)


class MeanDice(_SegmentationMetric):
    def compute(self):
        intersection = np.diag(self.confusion)
        sums = self.confusion.sum(1) + self.confusion.sum(0)
        dice = self._safe_divide(2 * intersection, sums)
        return np.nanmean(dice)
