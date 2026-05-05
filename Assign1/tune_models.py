import numpy as np
import torch
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
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
LEARNING_RATES = [1e-4, 1e-3] #Tune

os.makedirs("results", exist_ok=True)
mat = sio.loadmat("Xtrain.mat")
X_raw = np.array(mat["Xtrain"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, 1))

if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)

resultsRNN = []
resultsLSTM = []

best_rnn_score = float("inf")
best_lstm_score = float("inf")

best_rnn_config = None
best_lstm_config = None

best_rnn_model_state = None
best_lstm_model_state = None

#Iterate over every hyperparameter combination
for MODEL_CLASS in [RNN, LSTM]:
    for HIDDEN_DIM in HIDDEN_DIMS:
        for WINDOW_SIZE in WINDOW_SIZES:
            for LEARNING_RATE in LEARNING_RATES:
                laser_dataset = LaserDataset(X_scaled, WINDOW_SIZE)

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

                #Save MSE/MAE from the last epoch
                final_train_mse = history["train_mse"][-1]
                final_val_mse = history["val_mse"][-1]
                final_val_mae = history["val_mae"][-1]

                # Calculates a score where MSE weights twice as much than MAE to decide the best hyperparameter combo
                score = round(final_val_mse + 0.5 * final_val_mae, 7)

                config = {
                    "model": MODEL_CLASS.__name__,
                    "hidden_dim": HIDDEN_DIM,
                    "window_size": WINDOW_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "score": score,
                    "train_mse": final_train_mse,
                    "val_mse": final_val_mse,
                    "val_mae": final_val_mae,
                }
                # Keep only the best model for RNN and LSTM based of score and update results
                if MODEL_CLASS == RNN:
                    resultsRNN.append(config)
                    if score < best_rnn_score:
                        best_rnn_score = score
                else:
                    resultsLSTM.append(config)
                    if score < best_lstm_score:
                        best_lstm_score = score

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


