import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd

from utils.utils_data import distort_images, resize_crop
from utils.utils import PROJECT_ROOT


class KADIS700Dataset(Dataset):
    """
    KADIS700 dataset class used for pre-training the encoders for IQA.

    Args:
        root (string): root directory of the dataset
        patch_size (int): size of the patches to extract from the images
        max_distortions (int): maximum number of distortions to apply to the images
        num_levels (int): number of levels of distortion to apply to the images
        pristine_prob (float): probability of not distorting the images

    Returns:
        dictionary with keys:
            img_A_orig (Tensor): first view of the image pair
            img_A_ds (Tensor): downsampled version of the first view of the image pair (scale factor 2)
            img_B_orig (Tensor): second view of the image pair
            img_B_ds (Tensor): downsampled version of the second view of the image pair (scale factor 2)
            img_A_name (string): name of the image of the first view of the image pair
            img_B_name (string): name of the image of the second view of the image pair
            distortion_functions (list): list of the names of the distortion functions applied to the images
            distortion_values (list): list of the values of the distortion functions applied to the images
    """
    def __init__(self,
                 root: str,
                 patch_size: int = 224,
                 max_distortions: int = 4,
                 num_levels: int = 5,
                 pristine_prob: float = 0.05):

        root = Path(root)
        filenames_csv_path = PROJECT_ROOT / "data" / "synthetic_filenames.csv"
        if not filenames_csv_path.exists():
            self._generate_filenames_csv(root, filenames_csv_path)
        df = pd.read_csv(filenames_csv_path)
        self.ref_images = df["Filename"].tolist()
        self.ref_images = [Path(img) for img in self.ref_images]

        self.patch_size = patch_size
        self.max_distortions = max_distortions
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_levels = num_levels
        self.pristine_prob = pristine_prob

        assert 0 <= self.max_distortions <= 7, "The parameter max_distortions must be in the range [0, 7]"
        assert 1 <= self.num_levels <= 5, "The parameter num_levels must be in the range [1, 5]"

    def __getitem__(self, index: int) -> dict:
        img_A = Image.open(self.ref_images[index]).convert("RGB")
        img_A_name = self.ref_images[index].stem
        img_A_orig = resize_crop(img_A, self.patch_size)
        img_A_ds = resize_crop(img_A, self.patch_size, downscale_factor=2)

        # Randomly sample another image from the dataset
        random_idx = random.randint(0, len(self.ref_images) - 1)
        img_B = Image.open(self.ref_images[random_idx]).convert("RGB")
        img_B_name = self.ref_images[random_idx].stem
        img_B_orig = resize_crop(img_B, self.patch_size)
        img_B_ds = resize_crop(img_B, self.patch_size, downscale_factor=2)

        img_A_orig = transforms.ToTensor()(img_A_orig)
        img_B_orig = transforms.ToTensor()(img_B_orig)
        img_A_ds = transforms.ToTensor()(img_A_ds)
        img_B_ds = transforms.ToTensor()(img_B_ds)

        distort_functions = []
        distort_values = []
        # Distort images with (1 - self.pristine_prob) probability
        if random.random() > self.pristine_prob and self.max_distortions > 0:
            img_A_orig, distort_functions, distort_values = distort_images(img_A_orig,
                                                                         max_distortions=self.max_distortions,
                                                                         num_levels=self.num_levels)
            img_A_ds, _, _ = distort_images(img_A_ds, distort_functions=distort_functions, distort_values=distort_values)
            img_B_orig, _, _ = distort_images(img_B_orig, distort_functions=distort_functions, distort_values=distort_values)
            img_B_ds, _, _ = distort_images(img_B_ds, distort_functions=distort_functions, distort_values=distort_values)

        img_A_orig = self.normalize(img_A_orig)
        img_B_orig = self.normalize(img_B_orig)
        img_A_ds = self.normalize(img_A_ds)
        img_B_ds = self.normalize(img_B_ds)

        # Pad to make the length of distort_functions and distort_values equal for all samples
        distort_functions = [f.__name__ for f in distort_functions]
        distort_functions += [""] * (self.max_distortions - len(distort_functions))
        distort_values += [torch.inf] * (self.max_distortions - len(distort_values))

        return {"img_A_orig": img_A_orig, "img_A_ds": img_A_ds, "img_B_orig": img_B_orig, "img_B_ds": img_B_ds, "img_A_name": img_A_name, "img_B_name": img_B_name, "distortion_functions": distort_functions, "distortion_values": distort_values}

    def __len__(self) -> int:
        return len(self.ref_images)

    def _generate_filenames_csv(self, root: Path, csv_path: Path) -> None:
        """
        Generates a CSV file with the filenames of the images for faster preprocessing.
        """
        images = list((root / "ref_imgs").glob("*.png"))
        df = pd.DataFrame(images, columns=["Filename"])
        df.to_csv(csv_path, index=False)
