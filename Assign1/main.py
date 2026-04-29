import numpy as np
import scipy.io as sio
from model import RNN

INPUT_DIM = 1
HIDDEN_DIM = 16
NUM_LAYERS = 1

mat = sio.loadmat("XTrain.mat")
print(mat["Xtrain"])
X = np.array(mat["Xtrain"])
print(X.shape)

rnn = RNN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, layer_dim=NUM_LAYERS)

print(rnn)

