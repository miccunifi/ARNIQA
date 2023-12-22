from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from typing import Tuple

from utils.utils_data import resize_crop, center_corners_crop


class IQADataset(Dataset):
    """
    Base IQA dataset class.

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
            img (Tensor): the center crop and the 4 corners of the image (5 x 3 x crop_size x crop_size)
            img_ds (Tensor): downsampled version of the image (scale factor 2)
            mos (float): mean opinion score of the whole image
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
        self.mos_type = mos_type
        self.mos_range = mos_range
        self.is_synthetic = is_synthetic

        self.phase = phase
        assert self.phase in ["train", "test", "val", "all"], "phase must be in ['train', 'test', 'val', 'all']"

        self.split_idx = split_idx
        self.splits = {}
        for split in ["train", "test", "val"]:
            self.splits[split] = np.load(self.root / "splits" / f"{split}.npy")

        self.crop_size = crop_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.images = []
        self.mos = []

    def __getitem__(self, index: int) -> dict:
        img = Image.open(self.images[index]).convert("RGB")
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]
        img = torch.stack(crops, dim=0)

        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_ds = torch.stack(crops_ds, dim=0)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        mos = self.mos[index]
        return {"img": img, "img_ds": img_ds, "mos": mos}

    def __len__(self) -> int:
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        """
        Get the indices of the images in the specified split and phase.

        Args:
            split (int): idx of the split to use
            phase (str): phase of the dataset. Must be in ['train', 'test', 'val']
        """
        return self.splits[phase][split]
