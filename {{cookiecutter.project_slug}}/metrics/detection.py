from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Literal
from metrics.metric import _Metric


class _DetectionMetric(_Metric):
    def __init__(self):
        super().__init__()
        self.metric = None

    def reset(self):
        self.metric.reset()

    def update(self, preds, targets):
        self.metric.update(preds=preds, target=targets)

    def compute(self):
        return self.metric.compute()


class MAP(_DetectionMetric):
    def __init__(
            self,
            box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",  # 可选 "xyxy", "xywh", "cxcywh"
            iou_type: Literal["bbox", "segm"] = "bbox",  # 可选 "bbox", "segm"
            iou_thresholds: list[float] = None,  # IoU 阈值列表
            rec_thresholds: list[float] = None,  # Recall 曲线采样点
            max_detection_thresholds=None,  # 每图最多预测框
            class_metrics: bool = False,  # 是否输出每类 mAP/mAR
            **kwargs
    ):
        super().__init__()
        self.metric = MeanAveragePrecision(
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            **kwargs
        )
