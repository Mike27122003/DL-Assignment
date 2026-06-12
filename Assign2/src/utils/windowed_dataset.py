import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    def __init__(self, matrices, window_size, stride, labels, downsample_rate):

        self.matrices = [torch.tensor(m, dtype=torch.float32) for m in matrices]
        self.window_size = max(window_size // downsample_rate, 1)
        stride = max(stride // downsample_rate, 1)
        self.window_indices = self._generate_window_indices(stride, labels)

    def _generate_window_indices(self, stride, labels):
        window_indices = []
        for chunk_idx, matrix in enumerate(self.matrices):
            T = matrix.shape[1]
            label = labels[chunk_idx]

            if T < self.window_size:
                continue

            for start in range(0, T - self.window_size + 1, stride):
                window_indices.append((chunk_idx, start, label))

        return np.array(window_indices)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        chunk_idx, start, label = self.window_indices[idx]
        window = self.matrices[chunk_idx][:, start : start + self.window_size]
        return window, torch.tensor(label, dtype=torch.long)
