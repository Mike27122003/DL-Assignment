from torch.utils.data import Dataset
from torch import FloatTensor


class LaserDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, index):
        x = self.data[index : index + self.window_size]
        y = self.data[index + self.window_size]
        return x, y
