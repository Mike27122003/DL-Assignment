import h5py as h5
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from CNN import CNN
import torch.nn as nn

WINDOW_SIZE = 500
STRIDE = 200
NUMBER_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
FOLDER = Path("preprocessed_data/10/Intra/train")

#Loads the actual data from 1 training file
def load_single_data(filename_path):
    with h5.File(filename_path, "r") as f:
        key = list(f.keys())[0]
        matrix = f[key][()]
        return matrix

#Iterates over all training files (Only Intra for now)
def load_all_data():
    files = list(FOLDER.glob("*.h5"))
    X_all = []
    y_all = []

    for file in files:
        filename = str(file)
        matrix = load_single_data(filename)
        windows = create_window(matrix)
        label = get_label(filename)
        y = np.full(len(windows), label)
        X_all.append(windows)
        y_all.append(y)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    return X, y

def get_label(filename):
    if "rest" in filename:
        return 0
    elif "task_motor" in filename:
        return 1
    elif "task_story_math" in filename:
        return 2
    elif "task_working_memory" in filename:
        return 3

def create_window(matrix):
    windows = []
    T = matrix.shape[1]

    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        window = matrix[:, start:start + WINDOW_SIZE]
        windows.append(window.astype(np.float32))
    return np.array(windows)

X, y = load_all_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(X.shape)
print(y.shape)

model = CNN(
    window_size=WINDOW_SIZE,
    conv_channels=[64, 128],
    kernel_sizes=[7, 5]
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)