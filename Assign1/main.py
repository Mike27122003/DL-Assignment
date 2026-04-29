import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from model import RNN
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

mat = sio.loadmat("Xtrain.mat")
X_raw = np.array(mat["Xtrain"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, 1))

rnn = RNN(
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
optimizer = optim.AdamW(rnn.parameters(), lr=LEARNING_RATE)

history = {"train_mse": [], "val_mse": [], "val_mae": []}

# Main Epoch Bar
epoch_pbar = tqdm(range(MAX_EPOCHS), desc="Training Progress", unit="epoch")

for epoch in epoch_pbar:
    rnn.train()
    running_train_mse = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = rnn(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        running_train_mse += loss.item()

    rnn.eval()
    running_val_mse = 0.0
    running_val_mae = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            val_preds = rnn(batch_x)

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


# Convert history to a DataFrame for easy viewing
history_df = pd.DataFrame(history)
print("\nFinal Training History Preview:")
print(history_df.tail())

plt.figure(figsize=(12, 5))

# Plot MSE
plt.subplot(1, 2, 1)
plt.plot(history["train_mse"], label="Train MSE")
plt.plot(history["val_mse"], label="Val MSE")
plt.title("Model MSE (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history["val_mae"], label="Val MAE", color="orange")
plt.title("Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Error (Scaled Units)")
plt.legend()

plt.tight_layout()
plt.show()
