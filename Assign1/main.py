import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from model import RNN, LSTM
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from dataset import LaserDataset
import matplotlib.pyplot as plt

INPUT_DIM = 1
HIDDEN_DIM = 32
OUTPUT_DIM = 1
NUM_LAYERS = 1
MAX_EPOCHS = 100
WINDOW_SIZE = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PREDICT_FUTURE = 200
USE_LSTM = False

mat = sio.loadmat("Xtrain.mat")
X_raw = np.array(mat["Xtrain"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, 1))

if USE_LSTM:
    model = LSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        layer_dim=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
    )
else:
    model = RNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        layer_dim=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
    )

laser_dataset = LaserDataset(X_scaled, WINDOW_SIZE)

train_size = int(0.8 * len(laser_dataset))

train_data = Subset(laser_dataset, range(train_size))
val_data = Subset(laser_dataset, range(train_size, len(laser_dataset)))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

history = {"train_mse": [], "val_mse": [], "val_mae": []}

# Main Epoch Bar
epoch_pbar = tqdm(range(MAX_EPOCHS), desc="Training Progress", unit="epoch")

for epoch in epoch_pbar:
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

    epoch_train_mse = running_train_mse / len(train_loader)
    epoch_val_mse = running_val_mse / len(val_loader)
    epoch_val_mae = running_val_mae / len(val_loader)

    history["train_mse"].append(epoch_train_mse)
    history["val_mse"].append(epoch_val_mse)
    history["val_mae"].append(epoch_val_mae)

    # Update the main progress bar with the latest stats
    epoch_pbar.set_postfix(
        {
            "Tr_MSE": f"{epoch_train_mse:.5f}",
            "Val_MSE": f"{epoch_val_mse:.5f}",
            "Val_MAE": f"{epoch_val_mae:.5f}",
        }
    )


# run the model to test
model.eval()
all_loader = DataLoader(laser_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Convert history to a DataFrame for easy viewing
history_df = pd.DataFrame(history)
print("\nFinal Training History Preview:")
print(history_df.tail())

# plots
plt.figure(figsize=(15, 10))

# MSE plot
plt.subplot(2, 2, 1)
plt.plot(history["train_mse"], label="Train MSE")
plt.plot(history["val_mse"], label="Val MSE")
plt.title("Model MSE (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# MAE plot
plt.subplot(2, 2, 2)
plt.plot(history["val_mae"], label="Val MAE", color="orange")
plt.title("Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Error (Scaled)")
plt.legend()

# Data plot
plt.subplot(2, 2, (3, 4))

# Plot actual data
time_actual = np.arange(len(actual_unscaled))
plt.plot(time_actual, actual_unscaled, label="Actual Data", color="black", alpha=0.5)

# Plot dataset predictions
time_in_sample = np.arange(WINDOW_SIZE, WINDOW_SIZE + len(in_sample_unscaled))
plt.plot(
    time_in_sample,
    in_sample_unscaled,
    label="Dataset predictions",
    color="blue",
    linestyle="--",
    alpha=0.8,
)

# Plot Future predictions
time_future = np.arange(len(actual_unscaled), len(actual_unscaled) + PREDICT_FUTURE)
plt.plot(
    time_future,
    future_unscaled,
    label=f"Future predictions ({PREDICT_FUTURE} steps)",
    color="red",
    linewidth=2,
)

# Vertical line to show where the real data ends
plt.axvline(
    x=len(actual_unscaled), color="green", linestyle=":", label="Forecast Start"
)

plt.title("Plotted data")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = "LSTM" if USE_LSTM else "RNN"
plt.savefig(f"plots/{plot_filename}.png")
plt.show()