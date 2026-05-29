import h5py as h5
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from CNN import CNN
import torch.nn as nn
from torch import tensor
from torchmetrics.classification import Accuracy


# Loads the actual data from 1 training file
def load_single_data(filename_path):
    with h5.File(filename_path, "r") as f:
        key = list(f.keys())[0]
        matrix = f[key][()]
        return matrix


# Iterates over all training files (Only Intra for now)
def load_all_data(FOLDER, window_size, stride):
    files = list(FOLDER.glob("*.h5"))
    X_all = []
    y_all = []

    for file in files:
        filename = str(file)
        matrix = load_single_data(filename)
        windows = create_window(matrix, window_size, stride)
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


def create_window(matrix, window_size, stride):
    windows = []
    T = matrix.shape[1]

    for start in range(0, T - window_size + 1, stride):
        window = matrix[:, start : start + window_size]
        windows.append(window.astype(np.float32))
    return np.array(windows)


def train_cnn(window_size, stride, number_epochs, batch_size, learning_rate, folder):
    X_train, y_train = load_all_data(folder, window_size, stride)
    X_train = tensor(X_train, dtype=torch.float32)
    y_train = tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CNN(window_size=window_size, conv_channels=[32, 64], kernel_sizes=[7, 5])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    X_test, y_test = load_all_data(folder, window_size, stride)
    X_test = tensor(X_test, dtype=torch.float32)
    y_test = tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    acc = Accuracy(task="multiclass", num_classes=4)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            _, preds = torch.max(model(batch_x), 1)
            acc.update(preds, batch_y)

    test_acc = acc.compute()
    print(f"Test Accuracy: {test_acc}")
