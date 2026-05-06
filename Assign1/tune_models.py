import numpy as np
import torch
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from plots import plot_history
from model import RNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from dataset import LaserDataset
import os
import json

RANDOM_SEED = 42  # for reproducability. Set to None to use full random process
INPUT_DIM = 1
OUTPUT_DIM = 1
NUM_LAYERS = 1
MAX_EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_DIMS = [32, 64] #Tune
WINDOW_SIZES = [10, 20, 30, 40] #Tune
LEARNING_RATES = [1e-3, 1e-4] #Tune

os.makedirs("results", exist_ok=True)
mat = sio.loadmat("Xtrain.mat")
X_raw = np.array(mat["Xtrain"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, 1))

resultsRNN = []
resultsLSTM = []

best_rnn_score = float("inf")
best_lstm_score = float("inf")

best_rnn_history = None
best_lstm_history = None

#Iterate over every hyperparameter combination
for MODEL_CLASS in [RNN, LSTM]:
    for WINDOW_SIZE in WINDOW_SIZES:
        laser_dataset = LaserDataset(X_scaled, WINDOW_SIZE)
        for HIDDEN_DIM in HIDDEN_DIMS:
            for LEARNING_RATE in LEARNING_RATES:
                if RANDOM_SEED is not None:
                    torch.manual_seed(RANDOM_SEED)
                train_size = int(0.8 * len(laser_dataset))
                train_data = Subset(laser_dataset, range(train_size))
                val_data = Subset(laser_dataset, range(train_size, len(laser_dataset)))

                train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

                model = MODEL_CLASS(
                    input_dim=INPUT_DIM,
                    hidden_dim=HIDDEN_DIM,
                    layer_dim=NUM_LAYERS,
                    output_dim=OUTPUT_DIM,
                )

                criterion = nn.MSELoss()
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

                history = {"train_mse": [], "val_mse": [], "val_mae": []}

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
                    running_val_mse = 0.0
                    running_val_mae = 0.0

                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            val_preds = model(batch_x)

                            v_mse = criterion(val_preds, batch_y)
                            v_mae = torch.abs(val_preds - batch_y).mean()

                            running_val_mse += v_mse.item()
                            running_val_mae += v_mae.item()

                    #Round to 7 decimals
                    epoch_train_mse = round(running_train_mse / len(train_loader), 7)
                    epoch_val_mse = round(running_val_mse / len(val_loader), 7)
                    epoch_val_mae = round(running_val_mae / len(val_loader), 7)

                    history["train_mse"].append(epoch_train_mse)
                    history["val_mse"].append(epoch_val_mse)
                    history["val_mae"].append(epoch_val_mae)

                #Select the epoch with the lowest score
                val_mse = np.array(history["val_mse"])
                val_mae = np.array(history["val_mae"])

                # Calculates score for each epoch where MSE weights twice as much than MAE to decide the best hyperparameter combo
                scores = val_mse + 0.5 * val_mae
                best_epoch_id = int(np.argmin(scores)) #Retrieves the epoch with the lowest score

                # Save MSE/MAE from the best epoch
                final_train_mse = history["train_mse"][best_epoch_id]
                final_val_mse = history["val_mse"][best_epoch_id]
                final_val_mae = history["val_mae"][best_epoch_id]
                final_score = round(float(scores[best_epoch_id]), 7) #Round score to 7 decimals

                config = {
                    "hidden_dim": HIDDEN_DIM,
                    "window_size": WINDOW_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "score": final_score,
                    "train_mse": final_train_mse, #Stored for debugging
                    "val_mse": final_val_mse, #Stored for debugging
                    "val_mae": final_val_mae, #Stored for debugging
                }
                # Keep only the best model for RNN and LSTM based of score and update results
                if MODEL_CLASS == RNN:
                    resultsRNN.append(config)
                    if final_score < best_rnn_score:
                        best_rnn_score = final_score
                        best_rnn_history = history.copy()
                else:
                    resultsLSTM.append(config)
                    if final_score < best_lstm_score:
                        best_lstm_score = final_score
                        best_lstm_history = history.copy()

#Order results by score
resultsRNN = sorted(resultsRNN, key=lambda x: x["score"])
resultsLSTM = sorted(resultsLSTM, key=lambda x: x["score"])

#Print all configs
print("RNN ordered by score")
for r in resultsRNN:
    print(r)

print("LSTM ordered by score")
for r in resultsLSTM:
    print(r)

#Print best configs
print("Best RNN\n:", resultsRNN[0])
print("Best LSTM\n:", resultsLSTM[0])

#Store results of all models and the best configs in \results
with open("results/rnn_configs.json", "w") as f:
    json.dump(resultsRNN, f, indent=4)

with open("results/lstm_configs.json", "w") as f:
    json.dump(resultsLSTM, f, indent=4)

with open("results/best_rnn.json", "w") as f:
    json.dump(resultsRNN[0], f, indent=4)

with open("results/best_lstm.json", "w") as f:
    json.dump(resultsLSTM[0], f, indent=4)

with open("results/best_rnn_history.json", "w") as f:
    json.dump(best_rnn_history, f, indent=4)

with open("results/best_lstm_history.json", "w") as f:
    json.dump(best_lstm_history, f, indent=4)

plot_history()


