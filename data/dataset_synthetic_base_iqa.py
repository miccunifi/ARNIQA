from pathlib import Path
from PIL import Image
import torch
from typing import Tuple
from torchvision import transforms

from utils.utils_data import resize_crop, center_corners_crop
from data.dataset_base_iqa import IQADataset


class SyntheticIQADataset(IQADataset):
    """
    Base IQA dataset class for synthetic datasets.

    Args:
        root (string): root directory of the dataset
        mos_type (string): indicates the type of MOS. Value must be in ['mos', 'dmos'], where 'mos' corresponds to Mean
                           Opinion Score (higher is better) and 'dmos' to Differential Mean Opinion Score (lower is better).
        mos_range (tuple): range of the MOS values. Default is (1, 100).
        is_synthetic (bool): indicates whether the dataset is synthetic or not. Default is False.
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        split_idx (int): index of the split to use between [0, 9]. Used only if phase != 'all'. Default is 0.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): image
            mos (float): mean opinion score of the image
            dist_type (string): type of distortion
            dist_group (string): distortion group that contains the given distortion type
            dist_level (int): level of distortion
    """
    def __init__(self,
                 root: str,
                 mos_type: str = "mos",
                 mos_range: Tuple[int, int] = (1, 100),
                 is_synthetic: bool = False,
                 phase: str = "train",
                 split_idx: int = 0,
                 crop_size: int = 224):
        self.root = Path(root)
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase, split_idx=split_idx, crop_size=crop_size)
        self.distortion_types = []
        self.distortion_groups = []
        self.distortion_levels = []
        self.ref_images = []

    def __getitem__(self, index: int) -> dict:
        ref_img = Image.open(self.ref_images[index]).convert("RGB")
        ref_img_ds = resize_crop(ref_img, crop_size=None, downscale_factor=2)

        crops = center_corners_crop(ref_img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]
        ref_img = torch.stack(crops, dim=0)
        crops_ds = center_corners_crop(ref_img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        ref_img_ds = torch.stack(crops_ds, dim=0)

        ref_img = self.normalize(ref_img)
        ref_img_ds = self.normalize(ref_img_ds)

        data = super().__getitem__(index)
        data["dist_type"] = self.distortion_types[index]
        data["dist_group"] = self.distortion_groups[index]
        data["dist_level"] = self.distortion_levels[index]
        data["ref_img"] = ref_img
        data["ref_img_ds"] = ref_img_ds
        return data
