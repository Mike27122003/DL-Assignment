import numpy as np
import h5py
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

INTRA_TRAIN_DIR = "Final Project data/Intra/train/"
INTRA_TEST_DIR = "Final Project data/Intra/test/"

CROSS_TRAIN_DIR = "Final Project data/Cross/train/"
CROSS_TEST_DIR1 = "Final Project data/Cross/test1/"
CROSS_TEST_DIR2 = "Final Project data/Cross/test2/"
CROSS_TEST_DIR3 = "Final Project data/Cross/test3/"

DIRECTORIES = [
    INTRA_TRAIN_DIR,
    # INTRA_TEST_DIR,
    # CROSS_TRAIN_DIR,
    # CROSS_TEST_DIR1,
    # CROSS_TEST_DIR2,
    # CROSS_TEST_DIR3
]

DOWNSAMPLE_SIZE = 10

scalar = StandardScaler() # Z-score normalization

def time_wise_normalize(M: np.ndarray):
    # TODO: This normalizes rows PER FILE. I thik we need to normalize over all data
    M_normalized = scalar.fit_transform(M.T).T # timewise (row) normalize
    return M_normalized

def process_dir(dir: str):
    for path in Path(dir).glob("*.h5"):
        print(f"Processing: {path.relative_to("Final Project Data")}")

        with h5py.File(path, "r") as f:
            dataset_name = "_".join(path.stem.split("_")[:-1]) # remove the last '_' and '.h5'
            M = f.get(dataset_name)[()]
            M = M[:, ::DOWNSAMPLE_SIZE]
            M = time_wise_normalize(M)
            print(M.shape)
            return M # Just for exploring (Remove this later)
            # TODO: Save pre-processed files

# for dir in DIRECTORIES:
#     process_dir(dir)

# === Code for visualizations below ===

filename = "rest_105923_1.h5"
path = INTRA_TRAIN_DIR + filename

for dir in DIRECTORIES:
    m = process_dir(dir)

# Heatmap of covariances between sensors
cov = np.cov(m)
plt.figure(figsize=(8, 8))
sns.heatmap(cov, cmap="coolwarm", center=0)
plt.title("Sensor Covariance")
plt.show()

# HEatmap of T values
plt.figure(figsize=(8, 8))
sns.heatmap(m, cmap="coolwarm", center=0)
plt.title("MEG Heatmap")
plt.show()