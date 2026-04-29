import numpy as np
import pandas as pd
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model import RNN
from tqdm import tqdm

INPUT_DIM = 1
HIDDEN_DIM = 16
NUM_LAYERS = 1
MAX_EPOCHS = 100

mat = sio.loadmat("XTrain.mat")
print(mat["Xtrain"])
X = np.array(mat["Xtrain"])
print(X.shape)

rnn = RNN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, layer_dim=NUM_LAYERS)

# TODO: make the train and val splits

criterion = nn.MSELoss()
optimizer = optim.AdamW(rnn.parameters(), lr=1e-3)

best_val_loss = float("inf")
epoch_pbar = tqdm(range(MAX_EPOCHS), desc=f"Split Epochs", unit="epoch")

train_losses = []
val_losses = []

for epoch in MAX_EPOCHS:
    # Training stuff
    rnn.train()
    train_loss = 0.0
    pass

    # Validation stuff
    rnn.eval()
    val_loss = 0.0
    pass

# Testing & Evaluate
# Use MSE and MAE
# Make a pd.DataFrame
test_loss = 0.0
all_preds = []
all_targets = []
pass
