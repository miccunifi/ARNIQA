from pathlib import Path
import pandas as pd
import numpy as np

from data.dataset_synthetic_base_iqa import SyntheticIQADataset


class LIVEDataset(SyntheticIQADataset):
    """
    LIVE IQA dataset with DMOS in range [1, 100].

    Args:
        root (string): root directory of the dataset
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        split_idx (int): index of the split to use between [0, 9]. Used only if phase != 'all'. Default is 0.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): image
            mos (float): differential mean opinion score of the image (in range [1, 100])
            dist_type (string): type of distortion
            dist_group (string): distortion group which contains the distortion type
            dist_level (int): level of distortion
    """
    def __init__(self,
                 root: str,
                 phase: str = "train",
                 split_idx: int = 0,
                 crop_size: int = 224):
        mos_type = "dmos"
        mos_range = (1, 100)
        is_synthetic = True
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase, split_idx=split_idx, crop_size=crop_size)
        scores_csv = pd.read_csv(self.root / "LIVE.txt", sep=",")

        self.images = scores_csv["dis_img_path"].values.tolist()
        self.images = [Path(el) for el in self.images]
        self.images = np.array([self.root / el.relative_to(el.parts[0]) for el in self.images])

        self.ref_images = scores_csv["ref_img_path"].values.tolist()
        self.ref_images = [Path(el) for el in self.ref_images]
        self.ref_images = np.array([self.root / el.relative_to(el.parts[0]) for el in self.ref_images])

        self.mos = np.array(scores_csv["score"].values.tolist())

        self.distortion_types = scores_csv["dis_type"].values.tolist()
        self.distortion_types = np.array([distortion_types_mapping[el] for el in self.distortion_types])
        self.distortion_groups = np.array([available_distortions[el] for el in self.distortion_types])
        self.distortion_levels = np.array([5] * len(self.distortion_types))

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[self.split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # Remove the padding (i.e. -1 indexes)
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]
            self.distortion_types = self.distortion_types[split_idxs]
            self.distortion_groups = self.distortion_groups[split_idxs]
            self.distortion_levels = self.distortion_levels[split_idxs]


distortion_types_mapping = {
    "jp2k": "jpeg2000",
    "jpeg": "jpeg",
    "wn": "white_noise",
    "gblur": "gaussian_blur",
    "fastfading": "fast_fading",
}

available_distortions = {
    "white_noise": "noise",
    "gaussian_blur": "blur",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "fast_fading": "jpeg",
}

distortion_groups = {
    "noise": ["white_noise"],
    "blur": ["gaussian_blur"],
    "jpeg": ["jpeg2000", "jpeg", "fast_fading"],
}
