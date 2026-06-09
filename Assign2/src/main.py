import argparse
from cnn_methods import train_cnn, test_cnn
from utils.dirs import ClassificationType, INTRA_SPLITS, CROSS_SPLITS, get_dir
from preprocess import pre_process_data

STRIDE = 400
NUMBER_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DOWNSAMPLE_SIZES = [10]
results = []

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

        results = []

        for WINDOW_SIZE in WINDOW_SIZES:
            print(f"Window Size: {WINDOW_SIZE}")
            model = train_cnn(
                WINDOW_SIZE,
                STRIDE,
                NUMBER_EPOCHS,
                BATCH_SIZE,
                LEARNING_RATE,
                BASE_WEIGHT_DECAY,
                BASE_DROPOUT_RATE,
                training_data_dir,
            )

            accs = []
            for testing_split in testing_splits:
                testing_data_dir = get_dir(
                    classification_type, testing_split, downsample_size=downsample_size
                )
                print(f"Testing on {testing_split.value} split...")
                acc = test_cnn(model, WINDOW_SIZE, STRIDE, BATCH_SIZE, testing_data_dir)
                accs.append(acc)
            mean_acc = sum(accs) / len(accs)
            results.append((WINDOW_SIZE, mean_acc))

        #Stores the window size that provides the highest avg acc across the 3 tests
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        BEST_WINDOW_SIZE = results_sorted[0][0]

        results = []
        for WEIGHT_DECAY in WEIGHT_DECAYS:
            for DROPOUT_RATE in DROPOUT_RATES:
                print(f"\nWindow={BEST_WINDOW_SIZE}, Weight Decay={WEIGHT_DECAY}, Dropout Rate={DROPOUT_RATE}")

                model = train_cnn(
                    BEST_WINDOW_SIZE,
                    STRIDE,
                    NUMBER_EPOCHS,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    WEIGHT_DECAY,
                    DROPOUT_RATE,
                    training_data_dir,
                )

                accs = []
                for testing_split in testing_splits:
                    testing_data_dir = get_dir(
                        classification_type, testing_split, downsample_size=downsample_size
                    )
                    acc = test_cnn(model, BEST_WINDOW_SIZE, STRIDE, BATCH_SIZE, testing_data_dir)
                    accs.append(acc)
                mean_acc = sum(accs) / len(accs)
                results.append({
                    "window_size": BEST_WINDOW_SIZE,
                    "weight_decay": WEIGHT_DECAY,
                    "dropout": DROPOUT_RATE,
                    "accs": accs,
                    "mean_acc": mean_acc
                })

        #Prints the accuracy and hyperparameter combo with the highest avg accuracy and the acc of each test
        results_sorted = sorted(results, key=lambda x: x["mean_acc"], reverse=True)
        print(results_sorted[0])

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
