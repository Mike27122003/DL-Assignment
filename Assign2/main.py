import numpy
import h5py

def get_dataset_name(f: str):
    f_without_dir = f.split("/")[-1]
    print(f_without_dir)
    temp = f_without_dir.split("_")[:-1]
    print(temp)
    dataset_name = "_".join(temp)
    print(dataset_name)
    return dataset_name

INTRA_TRAIN_DIR = "Final Project data/Intra/train/"
INTRA_TEST_DIR = "Final Project data/Intra/test/"
# Assign2\Final Project data\Intra\train\rest_105923_1.h5

filename = "rest_105923_1.h5"
path = INTRA_TRAIN_DIR + filename

with h5py.File(path, "r") as f:
    dataset_name = get_dataset_name(path)
    print(dataset_name)
    matrix = f.get(dataset_name)[()]
    print(type(matrix))
    print(matrix.shape)