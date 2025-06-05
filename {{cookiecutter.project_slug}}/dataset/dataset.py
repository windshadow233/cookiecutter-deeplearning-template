from torch.utils.data import Dataset as D
from torch.utils.data._utils.collate import default_collate


class Dataset(D):
    def __init__(self):
        super().__init__()

    def __len__(self):
        ...

    def __getitem__(self, item):
        """
        Retrieve an item from the dataset.
        :param item: index
        :return: return a dict by default
        """
        ...

    def data_collate_fn(self, batch):
        """
        Custom collate function to handle the batching of data.
        This function should be overridden in subclasses to provide
        specific batching logic.
        """
        return default_collate(batch)