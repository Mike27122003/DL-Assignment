import numpy
import h5py
from pathlib import Path

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

def process_dir(dir):
    for path in Path(dir).glob("*.h5"):
        print(f"Processing: {path.relative_to("Final Project Data")}")

        with h5py.File(path, "r") as f:
            dataset_name = "_".join(path.stem.split("_")[:-1]) # remove the last '_' and '.h5'
            matrix = f.get(dataset_name)[()]
            # print(matrix.shape)

for dir in DIRECTORIES:
    process_dir(dir)