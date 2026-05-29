import numpy as np
from utils.dirs import ClassificationType, Split, get_dir
from utils.file_loader import FileLoader


def z_normalize_timewise(M: np.ndarray):
    """
    Normalize an ndarray using the Z = (X - mu_t) / sigma_t formula.
    """
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    return (M - mean) / std


def downsample_data(M: np.ndarray, downsample_size: int):
    """
    Downsample an ndarray by selecting every nth column, where n is the downsample size.
    """
    return M[:, ::downsample_size]


def pre_process_data(
    downsample_size: int,
    classification_type: ClassificationType,
    split: Split,
):
    """
    Load data from the specified directory, normalize it using Z-score normalization, and downsample it.
    """
    input_dir = get_dir(classification_type, split)
    no_downsample_dir = get_dir(classification_type, split, downsample_size=1)
    output_dir = get_dir(classification_type, split, downsample_size=downsample_size)

    # if data exists, skip pre-processing
    if list(output_dir.glob("*.h5")):
        print(
            f"Preprocessed files already exist in {output_dir}. Skipping pre-processing."
        )
        return

    # check if z normalization is necessary (if no downsampled files exist, we need to z-normalize before downsampling)
    if not list(no_downsample_dir.glob("*.h5")):
        file_loader = FileLoader(input_dir, no_downsample_dir)
        for matrix, input_file_path in file_loader:
            matrix = z_normalize_timewise(matrix)
            file_loader.save_file(matrix, input_file_path.name)

    file_loader = FileLoader(no_downsample_dir, output_dir)
    for matrix, input_file_path in file_loader:
        matrix = downsample_data(matrix, downsample_size)
        file_loader.save_file(matrix, input_file_path.name)


if __name__ == "__main__":
    DOWNSAMPLE_SIZE = 10
    pre_process_data(DOWNSAMPLE_SIZE, ClassificationType.INTRA, Split.TRAIN)
