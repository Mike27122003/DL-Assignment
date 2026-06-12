import argparse
from cnn_methods import train_cnn, test_cnn
from utils.dirs import ClassificationType, INTRA_SPLITS, CROSS_SPLITS, get_dir
from preprocess import pre_process_data

TOTAL_UPDATES = 2000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DOWNSAMPLE_SIZES = [10, 50, 100, 150]
results = []

STRIDES = [20, 50, 100, 200, 300, 400, 500]
WINDOW_SIZES = [250, 500, 1000, 2000]
WEIGHT_DECAYS = [0, 1e-4, 1e-5]
DROPOUT_RATES = [0, 0.2, 0.5]
BASE_WEIGHT_DECAY = 1e-4
BASE_DROPOUT_RATE = 0.2


def main(classification_type: ClassificationType):
    if classification_type == ClassificationType.INTRA:
        splits = INTRA_SPLITS
    else:
        splits = CROSS_SPLITS

    total_results = []

    for downsample_size in DOWNSAMPLE_SIZES:
        print(f"Running training/testing for downsample size: {downsample_size}")

        print(f"Pre-processing {classification_type.value} data...")
        for split in splits:
            print(f"Pre-processing {split.value} split...")
            pre_process_data(downsample_size, classification_type, split)

        training_split = splits[0]
        testing_splits = splits[1:]

        training_data_dir = get_dir(
            classification_type, training_split, downsample_size=downsample_size
        )

        window_results = []

        for window_size in WINDOW_SIZES:
            for stride in STRIDES:
                print(f"Window Size: {window_size}")

                accs, mean_acc = train_test(
                    window_size=window_size,
                    stride=stride,
                    weight_decay=BASE_WEIGHT_DECAY,
                    dropout_rate=BASE_DROPOUT_RATE,
                    training_data_dir=training_data_dir,
                    downsample_size=downsample_size,
                    testing_splits=testing_splits,
                    classification_type=classification_type,
                )
                if mean_acc is not None:
                    window_results.append((window_size, stride, mean_acc))

        if len(window_results) == 0:
            print(
                f"No valid windows sizes for downsample size {downsample_size}, skipping hyperparameter tuning..."
            )
            continue

        # Stores the window size that provides the highest avg acc across the 3 tests
        results_sorted = sorted(window_results, key=lambda x: x[2], reverse=True)
        best_window_size = results_sorted[0][0]
        best_stride = results_sorted[0][1]

        for weight_decay in WEIGHT_DECAYS:
            for dropout_rate in DROPOUT_RATES:
                print(
                    f"\nWindow={best_window_size}, Weight Decay={weight_decay}, Dropout Rate={dropout_rate}"
                )

                accs, mean_acc = train_test(
                    window_size=best_window_size,
                    stride=best_stride,
                    weight_decay=weight_decay,
                    dropout_rate=dropout_rate,
                    training_data_dir=training_data_dir,
                    downsample_size=downsample_size,
                    testing_splits=testing_splits,
                    classification_type=classification_type,
                )
                total_results.append(
                    {
                        "downsample_size": downsample_size,
                        "window_size": best_window_size,
                        "stride": best_stride,
                        "weight_decay": weight_decay,
                        "dropout": dropout_rate,
                        "accs": accs,
                        "mean_acc": mean_acc,
                    }
                )

    # Prints the accuracy and hyperparameter combo with the highest avg accuracy and the acc of each test
    results_sorted = sorted(total_results, key=lambda x: x["mean_acc"], reverse=True)
    print(results_sorted[0])


def train_test(
    window_size,
    stride,
    weight_decay,
    dropout_rate,
    training_data_dir,
    downsample_size,
    testing_splits,
    classification_type,
):
    model = train_cnn(
        window_size,
        stride,
        TOTAL_UPDATES,
        BATCH_SIZE,
        LEARNING_RATE,
        weight_decay,
        dropout_rate,
        training_data_dir,
    )

    if model is None:
        print(
            f"No valid windows for window size {window_size} and downsample size {downsample_size}, skipping..."
        )
        return None, None

    accs = []
    for testing_split in testing_splits:
        testing_data_dir = get_dir(
            classification_type, testing_split, downsample_size=downsample_size
        )
        print(f"Testing on {testing_split.value} split...")
        acc = test_cnn(model, window_size, stride, BATCH_SIZE, testing_data_dir)
        accs.append(acc)
    mean_acc = sum(accs) / len(accs)
    return accs, mean_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEG classification model.")
    parser.add_argument(
        "--type",
        type=ClassificationType,
        choices=list(ClassificationType),
        required=True,
        help="The classification type to use (Intra or Cross).",
    )
    args = parser.parse_args()
    main(args.type)
