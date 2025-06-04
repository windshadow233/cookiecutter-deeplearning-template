from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        ...

    def __getitem__(self, item):
        ...