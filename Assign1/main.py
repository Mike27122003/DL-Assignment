import numpy as np
import scipy.io as sio

mat = sio.loadmat("XTrain.mat")
print(mat["Xtrain"])
X = np.array(mat["Xtrain"])
print(X.shape)
