import h5py as h5
import numpy as np
import torch
from torch.utils.data import DataLoader
from CNN import CNN
import torch.nn as nn
from torchmetrics.classification import Accuracy

from utils.windowed_dataset import WindowedDataset


# Loads the actual data from 1 training file
def load_single_data(filename_path):
    with h5.File(filename_path, "r") as f:
        key = list(f.keys())[0]
        matrix = f[key][()]
        return matrix


# Iterates over all training files
def load_all_data(FOLDER):
    files = list(FOLDER.glob("*.h5"))
    matrices = []
    labels = []

    for file in files:
        filename = str(file)
        matrix = load_single_data(filename)
        label = get_label(filename)

        matrices.append(matrix)
        labels.append(label)

    return matrices, labels


def get_label(filename):
    if "rest" in filename:
        return 0
    elif "task_motor" in filename:
        return 1
    elif "task_story_math" in filename:
        return 2
    elif "task_working_memory" in filename:
        return 3


def create_window(matrix, window_size, stride):
    windows = []
    T = matrix.shape[1]

    for start in range(0, T - window_size + 1, stride):
        window = matrix[:, start : start + window_size]
        windows.append(window.astype(np.float32))
    return np.array(windows)


def train_cnn(
    window_size,
    stride,
    total_updates,
    batch_size,
    learning_rate,
    weight_decay,
    dropout_rate,
    folder,
):
    matrices_train, labels_train = load_all_data(folder)
    matrices_train = [
        matrix for matrix in matrices_train if matrix.shape[1] >= window_size
    ]
    if len(matrices_train) == 0:
        return None
    train_dataset = WindowedDataset(matrices_train, window_size, stride, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CNN(
        window_size=window_size,
        dropout_rate=dropout_rate,
        conv_channels=[32, 64],
        kernel_sizes=[5, 3],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    number_epochs = total_updates * batch_size // len(train_dataset) + 1
    for epoch in range(number_epochs):
        running_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{number_epochs}], Loss: {running_loss/len(train_loader):.10f}"
        )

    return model


def test_cnn(model, window_size, stride, batch_size, folder):
    matrices_test, labels_test = load_all_data(folder)
    test_dataset = WindowedDataset(matrices_test, window_size, stride, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    acc = Accuracy(task="multiclass", num_classes=4)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            _, preds = torch.max(model(batch_x), 1)
            acc.update(preds, batch_y)

    test_acc = acc.compute().item()
    print(f"Test Accuracy: {test_acc}")
    return test_acc
