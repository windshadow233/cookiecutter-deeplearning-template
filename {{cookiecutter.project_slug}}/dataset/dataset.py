from torch.utils.data import Dataset as D
import torch


class Dataset(D):
    def __init__(self):
        super().__init__()

    def __len__(self):
        ...

    def __getitem__(self, item):
        ...

    def data_collate_fn(self, batch):
        """
        Custom collate function to handle the batching of data.
        This function should be overridden in subclasses to provide
        specific batching logic.
        """
        data, label = zip(*batch)
        data = torch.stack(data, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return {'x': data, 'y': label}