import argparse
from pathlib import Path
from cnn_methods import train_cnn, test_cnn
from utils.dirs import ClassificationType, INTRA_SPLITS, CROSS_SPLITS
from preprocess import pre_process_data

WINDOW_SIZE = 500
STRIDE = 400
NUMBER_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
INTRA_TRAIN_FOLDER = Path("preprocessed_data/10/Intra/train")
INTRA_TEST_FOLDER = Path("preprocessed_data/10/Intra/test")
CROSS_TRAIN_FOLDER = Path("preprocessed_data/10/Cross/train")
CROSS_TEST1_FOLDER = Path("preprocessed_data/10/Cross/test1")
CROSS_TEST2_FOLDER = Path("preprocessed_data/10/Cross/test2")
CROSS_TEST3_FOLDER = Path("preprocessed_data/10/Cross/test3")

def main(classification_type: ClassificationType):
    if classification_type == ClassificationType.INTRA:
        splits = INTRA_SPLITS
    else:
        splits = CROSS_SPLITS

    print(f"Pre-processing {classification_type.value} data...")
    for split in splits:
        print(f"Pre-processing {split.value} split...")
        pre_process_data(classification_type, split)

    if classification_type == ClassificationType.INTRA:
        model = train_cnn(WINDOW_SIZE, STRIDE, NUMBER_EPOCHS, BATCH_SIZE, LEARNING_RATE, INTRA_TRAIN_FOLDER)
        test_cnn(model, WINDOW_SIZE, STRIDE, BATCH_SIZE, INTRA_TEST_FOLDER)
    else:
        model = train_cnn(WINDOW_SIZE, STRIDE, NUMBER_EPOCHS, BATCH_SIZE, LEARNING_RATE, CROSS_TRAIN_FOLDER)
        print("test1 results")
        test_cnn(model, WINDOW_SIZE, STRIDE, BATCH_SIZE, CROSS_TEST1_FOLDER)
        print("test2 results")
        test_cnn(model, WINDOW_SIZE, STRIDE, BATCH_SIZE, CROSS_TEST2_FOLDER)
        print("test3 results")
        test_cnn(model, WINDOW_SIZE, STRIDE, BATCH_SIZE, CROSS_TEST3_FOLDER)

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
