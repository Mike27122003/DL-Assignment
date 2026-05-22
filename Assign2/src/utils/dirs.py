from enum import Enum
from pathlib import Path


class ClassificationType(Enum):
    INTRA = "Intra"
    CROSS = "Cross"


class Split(Enum):
    TRAIN = "train"
    TEST = "test"
    TEST1 = "test1"
    TEST2 = "test2"
    TEST3 = "test3"


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTRA_SPLITS = [Split.TRAIN, Split.TEST]
CROSS_SPLITS = [Split.TRAIN, Split.TEST1, Split.TEST2, Split.TEST3]


def get_dir(
    classification_type: ClassificationType,
    split: Split,
    *,
    downsample_size: int | None = None,
):
    if downsample_size is not None:
        return PROJECT_ROOT / Path(
            f"preprocessed_data/{downsample_size}/{classification_type.value}/{split.value}/"
        )
    else:
        return PROJECT_ROOT / Path(
            f"Final Project data/{classification_type.value}/{split.value}/"
        )
