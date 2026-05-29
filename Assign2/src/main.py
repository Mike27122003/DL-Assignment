import argparse
from cnn_methods import train_cnn, test_cnn
from utils.dirs import ClassificationType, INTRA_SPLITS, CROSS_SPLITS, get_dir
from preprocess import pre_process_data

WINDOW_SIZE = 500
STRIDE = 400
NUMBER_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DOWNSAMPLE_RATE = 1


def main(classification_type: ClassificationType):
    if classification_type == ClassificationType.INTRA:
        splits = INTRA_SPLITS
    else:
        splits = CROSS_SPLITS

    print(f"Pre-processing {classification_type.value} data...")
    for split in splits:
        print(f"Pre-processing {split.value} split...")
        pre_process_data(classification_type, split)

    training_split = splits[0]
    testing_splits = splits[1:]

    training_data_dir = get_dir(
        classification_type, training_split, downsample_size=DOWNSAMPLE_RATE
    )
    model = train_cnn(
        WINDOW_SIZE, STRIDE, NUMBER_EPOCHS, BATCH_SIZE, LEARNING_RATE, training_data_dir
    )

    for testing_split in testing_splits:
        testing_data_dir = get_dir(
            classification_type, testing_split, downsample_size=DOWNSAMPLE_RATE
        )
        print(f"Testing on {testing_split.value} split...")
        test_cnn(model, WINDOW_SIZE, STRIDE, BATCH_SIZE, testing_data_dir)


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
