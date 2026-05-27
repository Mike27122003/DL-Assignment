from utils.file_loader import FileLoader
from utils.dirs import ClassificationType, Split, get_dir
import numpy as np
import matplotlib.pyplot as plt
import math

"""
This script helps to find a good range of dropout rates to use, while still preserving the expressive power of the data.
The script will have the following steps:
1. Find good features to visualize
2. Test these features with different dropout rates and visualize the results.
3. Find good dropout rates to use for the final model.
"""


def phase1():
    """
    Find good features to visualize.
    """
    input_dir = get_dir(ClassificationType.INTRA, Split.TRAIN, downsample_size=1)
    file_loader = FileLoader(input_dir, input_dir)
    matrix, _ = next(iter(file_loader))

    variances = np.var(matrix, axis=1)
    good_feature_indices = np.argsort(variances)[-5:]

    return good_feature_indices, matrix


def phase2(feature_indices, matrix):
    """
    Test these features with different dropout rates and visualize the results.
    """
    downsample_sizes = [2, 3, 5, 7, 10]
    n_timestamps = 1000
    original_x = np.arange(n_timestamps)

    total_plots = len(downsample_sizes) + 1
    num_cols = math.ceil(math.sqrt(total_plots))
    num_rows = math.floor(math.sqrt(total_plots))

    for feature_idx in feature_indices:
        # extract feature and select timestamp
        original_feature_data = matrix[feature_idx, :n_timestamps]

        # Create 1 row with 4 columns side-by-side
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharey=True
        )
        fig.suptitle(
            f"Dropout Comparison for Feature Row Index: {feature_idx}",
            fontsize=14,
            fontweight="bold",
        )
        axes = axes.flatten()

        # 1. Plot Original Data
        axes[0].plot(original_x, original_feature_data, color="black", alpha=0.8)
        axes[0].set_title("Original dropout 1 (No dropout)")
        axes[0].set_xlabel("Timestamps")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Plot Dropout variations side-by-side
        for i, downsample_size in enumerate(downsample_sizes):
            ax = axes[i + 1]

            sampled_data = original_feature_data[::downsample_size]
            sampled_ticks = original_x[::downsample_size]

            ax.plot(sampled_ticks, sampled_data, color="crimson", alpha=0.7)
            ax.set_title(f"Dropout size {downsample_size}")
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    feature_indices, matrix = phase1()
    phase2(feature_indices, matrix)
