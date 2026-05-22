import argparse

from utils.dirs import ClassificationType, INTRA_SPLITS, CROSS_SPLITS
from preprocess import pre_process_data


def main(classification_type: ClassificationType):
    if classification_type == ClassificationType.INTRA:
        splits = INTRA_SPLITS
    else:
        splits = CROSS_SPLITS

    print(f"Pre-processing {classification_type.value} data...")
    for split in splits:
        print(f"Pre-processing {split.value} split...")
        pre_process_data(classification_type, split)


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
