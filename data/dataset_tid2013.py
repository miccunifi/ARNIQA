import pandas as pd
import re
import numpy as np

from data.dataset_synthetic_base_iqa import SyntheticIQADataset


class TID2013Dataset(SyntheticIQADataset):
    """
    TID2013 IQA dataset with MOS in range [0, 9].

    Args:
        root (string): root directory of the dataset
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        split_idx (int): index of the split to use between [0, 9]. Used only if phase != 'all'. Default is 0.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): image
            mos (float): differential mean opinion score of the image (in range [0, 9])
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
        mos_range = (0, 9)
        is_synthetic = True
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase, split_idx=split_idx, crop_size=crop_size)
        scores_csv = pd.read_csv(self.root / "mos_with_names.txt", sep=" ", header=None, names=["mos", "img_name"])

        self.images = scores_csv["img_name"].values.tolist()
        self.ref_images = [el.split("_")[0].upper() + ".BMP" for el in self.images]
        self.images = np.array([self.root / "distorted_images" / el for el in self.images])

        self.ref_images = np.array([self.root / "reference_images" / el for el in self.ref_images])
        self.ref_images = np.array([el if el.exists() else self.root / "reference_images" / el.name.lower() for el in self.ref_images])

        self.mos = np.array(scores_csv["mos"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[self.split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))      # Remove the padding (i.e. -1 indexes)
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]

        for image in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.bmp$', str(image), re.IGNORECASE)
            dist_type = distortion_types_mapping[int(match.group(1))]
            self.distortion_types.append(dist_type)
            self.distortion_groups.append(available_distortions[dist_type])
            self.distortion_levels.append(int(match.group(2)))
        self.distortion_types = np.array(self.distortion_types)
        self.distortion_groups = np.array(self.distortion_groups)
        self.distortion_levels = np.array(self.distortion_levels)


distortion_types_mapping = {
    1: "additive_gaussian_noise",
    2: "intensive_color_noise",
    3: "spatially_correlated_noise",
    4: "masked_noise",
    5: "high_frequency_noise",
    6: "impulse_noise",
    7: "quantization_noise",
    8: "gaussian_blur",
    9: "image_denoising",
    10: "jpeg_compression",
    11: "jpeg2000_compression",
    12: "jpeg_transmission_errors",
    13: "jpeg2000_transmission_errors",
    14: "non_eccentricity_pattern_noise",
    15: "local_blockwise_distortions",
    16: "mean_shift",
    17: "contrast_change",
    18: "color_saturation_change",
    19: "multiplicative_gaussian_noise",
    20: "comfort_noise",
    21: "lossy_compression_noisy_images",
    22: "color_quantization_with_dither",
    23: "chromatic_aberrations",
    24: "sparse_sampling_reconstruction"
}

available_distortions = {
    'gaussian_blur': 'blur',
    'intensive_color_noise': 'color_distortion',
    'color_saturation_change': 'color_distortion',
    'color_quantization_with_dither': 'color_distortion',
    'chromatic_aberrations': 'color_distortion',
    'jpeg_compression': 'jpeg',
    'jpeg2000_compression': 'jpeg',
    'jpeg_transmission_errors': 'jpeg',
    'jpeg2000_transmission_errors': 'jpeg',
    'lossy_compression_noisy_images': 'noise',
    'image_denoising': 'noise',
    'additive_gaussian_noise': 'noise',
    'spatially_correlated_noise': 'noise',
    'masked_noise': 'noise',
    'high_frequency_noise': 'noise',
    'impulse_noise': 'noise',
    'quantization_noise': 'noise',
    'multiplicative_gaussian_noise': 'noise',
    'comfort_noise': 'noise',
    'mean_shift': 'brightness_change',
    'local_blockwise_distortions': 'spatial_distortion',
    'non_eccentricity_pattern_noise': 'spatial_distortion',
    'sparse_sampling_reconstruction': 'spatial_distortion',
    'contrast_change': 'sharpness_contrast'
}

distortion_groups = {
    "blur": ["gaussian_blur"],
    "color_distortion": ["intensive_color_noise", "color_saturation_change", "color_quantization_with_dither", "chromatic_aberrations"],
    "jpeg": ["jpeg_compression", "jpeg2000_compression", "jpeg_transmission_errors", "jpeg2000_transmission_errors"],
    "noise": ["lossy_compression_noisy_images", "image_denoising", "additive_gaussian_noise", "spatially_correlated_noise", "masked_noise", "high_frequency_noise", "impulse_noise", "quantization_noise", "multiplicative_gaussian_noise", "comfort_noise"],
    "brightness_change": ["mean_shift"],
    "spatial_distortion": ["local_blockwise_distortions", "non_eccentricity_pattern_noise", "sparse_sampling_reconstruction"],
    "sharpness_contrast": ["contrast_change"]
}
