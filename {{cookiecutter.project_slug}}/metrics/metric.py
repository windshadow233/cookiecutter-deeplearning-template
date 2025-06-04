from abc import ABC, abstractmethod


class _Metric(ABC):
    @abstractmethod
    def update(self, **kwargs):
        ...

    @abstractmethod
    def compute(self, **kwargs):
        ...

    @abstractmethod
    def reset(self):
        ...


class AvgMetric(_Metric):
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count):
        self.total += float(value)
        self.count += count

    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0