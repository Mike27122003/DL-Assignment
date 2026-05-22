import h5py
from pathlib import Path
import numpy as np


class FileLoader:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_base_path = input_dir
        self.output_base_path = output_dir
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        self.files = sorted(list(self.input_base_path.glob("*.h5")))
        self.current_index = 0

    def get_dataset_name(self, f: Path):
        f_without_dir = f.name
        temp = f_without_dir.split("_")[:-1]
        dataset_name = "_".join(temp)
        return dataset_name

    def _load_h5_file(self, file_path: Path):
        with h5py.File(file_path, "r") as f:
            dataset_name = self.get_dataset_name(file_path)
            matrix = f.get(dataset_name)[()]
            return matrix

    def _save_h5_file(self, matrix, file_path: Path):
        with h5py.File(file_path, "w") as f:
            dataset_name = self.get_dataset_name(file_path)
            f.create_dataset(dataset_name, data=matrix, compression="gzip")

    def save_file(self, matrix: np.ndarray, file_name: str):
        file_path = self.output_base_path / file_name
        self._save_h5_file(matrix, file_path)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.files):
            raise StopIteration
        file_path = self.files[self.current_index]
        matrix = self._load_h5_file(file_path)
        self.current_index += 1
        return matrix, file_path
