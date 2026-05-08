import os

import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from model import RNN, LSTM
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from dataset import LaserDataset
import json
from plots import plot_forecasts
from evaluate import evaluate

RANDOM_SEED = 42
INPUT_DIM = 1
OUTPUT_DIM = 1
NUM_LAYERS = 1
BATCH_SIZE = 32
MAX_EPOCHS = 100
PREDICT_FUTURE = 200

with open("results/best_rnn.json", "r") as f:
    rnn_config = json.load(f)
with open("results/best_lstm.json", "r") as f:
    lstm_config = json.load(f)

HIDDEN_DIM_RNN = int(rnn_config["hidden_dim"])
WINDOW_SIZE_RNN = int(rnn_config["window_size"])
LEARNING_RATE_RNN = float(rnn_config["learning_rate"])
HIDDEN_DIM_LSTM = int(lstm_config["hidden_dim"])
WINDOW_SIZE_LSTM = int(lstm_config["window_size"])
LEARNING_RATE_LSTM = float(lstm_config["learning_rate"])

if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)

def run_model(model, training_data, WINDOW_SIZE, LEARNING_RATE):
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)



    for epoch in range(MAX_EPOCHS):
        model.train()
        running_train_mse = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_train_mse += loss.item()

    model.eval()

    all_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)

    # get predictions for existing dataset
    dataset_predictions = []

    with torch.no_grad():
        for batch_x, _ in all_loader:
            preds = model(batch_x)
            dataset_predictions.append(preds.numpy())

    dataset_predictions = np.concatenate(dataset_predictions, axis=0)

    # predict future recursively (take the last WINDOW_SIZE items)
    current_window = torch.tensor(X_scaled[-WINDOW_SIZE:]).view(1, WINDOW_SIZE, 1).float()
    future_predictions = []

    for _ in range(PREDICT_FUTURE):
        with torch.no_grad():
            pred = model(current_window)
            future_predictions.append(pred.item())

            # slide the window
            new_point = pred.view(1, 1, 1)
            current_window = torch.cat((current_window[:, 1:, :], new_point), dim=1)

    # scale values back
    in_sample_unscaled = scaler.inverse_transform(dataset_predictions.reshape(-1, 1))
    future_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    actual_unscaled = scaler.inverse_transform(X_scaled)

    return in_sample_unscaled, future_unscaled, actual_unscaled

mat = sio.loadmat("Xtrain.mat")
X_raw = np.array(mat["Xtrain"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, 1))

model_rnn = RNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM_RNN,
        layer_dim=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
    )

model_lstm= LSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM_LSTM,
        layer_dim=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
    )

training_data_rnn = LaserDataset(X_scaled, WINDOW_SIZE_RNN)
training_data_lstm = LaserDataset(X_scaled, WINDOW_SIZE_LSTM)

rnn_in, rnn_future, actual = run_model(model_rnn, training_data_rnn, WINDOW_SIZE_RNN, LEARNING_RATE_RNN)
lstm_in, lstm_future, _ = run_model(model_lstm, training_data_lstm, WINDOW_SIZE_LSTM, LEARNING_RATE_LSTM) #actual is already retrieved in RNN
plot_forecasts(actual, rnn_in, rnn_future, lstm_in, lstm_future, WINDOW_SIZE_RNN, WINDOW_SIZE_LSTM, PREDICT_FUTURE)
evaluate(rnn_future, lstm_future)
