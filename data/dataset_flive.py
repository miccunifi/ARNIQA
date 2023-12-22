import pandas as pd
import numpy as np

from data.dataset_base_iqa import IQADataset


class FLIVEDataset(IQADataset):
    """
    FLIVE IQA dataset with MOS in range [1, 100]. FLIVE has only the official split.

    Args:
        root (string): root directory of the dataset
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): image
            mos (float): mean opinion score of the image (in range [1, 100])
    """
    def __init__(self,
                 root: str,
                 phase: str = "train",
                 crop_size: int = 224):
        mos_type = "mos"
        mos_range = (1, 100)
        is_synthetic = False
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase, crop_size=crop_size)
        scores_csv = pd.read_csv(self.root / "labels_image.csv")

        self.images = scores_csv["name"].values.tolist()
        self.images = np.array([self.root / "database" / el for el in self.images])

        self.mos = np.array(scores_csv["mos"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # Remove the padding (i.e. -1 indexes)
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]

    def get_split_indices(self, split: int, phase: str):
        # The split argument is ignored because FLIVE has only the official split. Needed to be compatible with base class.
        return self.splits[phase]
