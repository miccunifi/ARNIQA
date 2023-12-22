import pandas as pd
import re
import numpy as np

from data.dataset_synthetic_base_iqa import SyntheticIQADataset


class KADID10KDataset(SyntheticIQADataset):
    """
    KADID-10k IQA dataset with MOS in range [1, 5].

    Args:
        root (string): root directory of the dataset
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        split_idx (int): index of the split to use between [0, 9]. Used only if phase != 'all'. Default is 0.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): image
            mos (float): differential mean opinion score of the image (in range [1, 5])
            dist_type (string): type of distortion
            dist_group (string): distortion group which contains the distortion type
            dist_level (int): level of distortion
    """
    def __init__(self,
                 root: str,
                 phase: str = "train",
                 split_idx: int = 0,
                 crop_size: int = 224):
        mos_type = "mos"
        mos_range = (1, 5)
        is_synthetic = True
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase, split_idx=split_idx, crop_size=crop_size)
        scores_csv = pd.read_csv(self.root / "dmos.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        self.images = scores_csv["dist_img"].values.tolist()
        self.images = np.array([self.root / "images" / el for el in self.images])

        self.ref_images = scores_csv["ref_img"].values.tolist()
        self.ref_images = np.array([self.root / "images" / el for el in self.ref_images])

        self.mos = np.array(scores_csv["dmos"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[self.split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # Remove the padding (i.e. -1 indexes)
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]

        for image in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(image))
            dist_type = distortion_types_mapping[int(match.group(1))]
            self.distortion_types.append(dist_type)
            self.distortion_groups.append(available_distortions[dist_type])
            self.distortion_levels.append(int(match.group(2)))
        self.distortion_types = np.array(self.distortion_types)
        self.distortion_groups = np.array(self.distortion_groups)
        self.distortion_levels = np.array(self.distortion_levels)


distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

available_distortions = {
    "gaussian_blur": "blur",
    "lens_blur": "blur",
    "motion_blur": "blur",
    "color_diffusion": "color_distortion",
    "color_shift": "color_distortion",
    "color_quantization": "color_distortion",
    "color_saturation_1": "color_distortion",
    "color_saturation_2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "white_noise": "noise",
    "white_noise_color_component": "noise",
    "impulse_noise": "noise",
    "multiplicative_noise": "noise",
    "denoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "mean_shift": "brightness_change",
    "jitter": "spatial_distortion",
    "non_eccentricity_patch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "color_block": "spatial_distortion",
    "high_sharpen": "sharpness_contrast",
    "contrast_change": "sharpness_contrast"
}

distortion_groups = {
    "blur": ["gaussian_blur", "lens_blur", "motion_blur"],
    "color_distortion": ["color_diffusion", "color_shift", "color_quantization", "color_saturation_1", "color_saturation_2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["white_noise", "white_noise_color_component", "impulse_noise", "multiplicative_noise", "denoise"],
    "brightness_change": ["brighten", "darken", "mean_shift"],
    "spatial_distortion": ["jitter", "non_eccentricity_patch", "pixelate", "quantization", "color_block"],
    "sharpness_contrast": ["high_sharpen", "contrast_change"]
}
