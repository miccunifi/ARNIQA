from random import randrange
import torchvision.transforms.functional as TF
from typing import List, Callable, Union
from PIL.Image import Image as PILImage

from utils.distortions import *


distortion_groups = {
    "blur": ["gaublur", "lensblur", "motionblur"],
    "color_distortion": ["colordiff", "colorshift", "colorsat1", "colorsat2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["whitenoise", "whitenoiseCC", "impulsenoise", "multnoise"],
    "brightness_change": ["brighten", "darken", "meanshift"],
    "spatial_distortion": ["jitter", "noneccpatch", "pixelate", "quantization", "colorblock"],
    "sharpness_contrast": ["highsharpen", "lincontrchange", "nonlincontrchange"],
}

distortion_groups_mapping = {
    "gaublur": "blur",
    "lensblur": "blur",
    "motionblur": "blur",
    "colordiff": "color_distortion",
    "colorshift": "color_distortion",
    "colorsat1": "color_distortion",
    "colorsat2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "whitenoise": "noise",
    "whitenoiseCC": "noise",
    "impulsenoise": "noise",
    "multnoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "meanshift": "brightness_change",
    "jitter": "spatial_distortion",
    "noneccpatch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "colorblock": "spatial_distortion",
    "highsharpen": "sharpness_contrast",
    "lincontrchange": "sharpness_contrast",
    "nonlincontrchange": "sharpness_contrast",
}

distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    "colordiff": [1, 3, 6, 8, 12],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    "colorsat2": [1, 2, 3, 6, 9],
    "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "noneccpatch": [20, 40, 60, 80, 100],
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "quantization": [20, 16, 13, 10, 7],
    "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],
    "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}

distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    "colordiff": color_diffusion,
    "colorshift": color_shift,
    "colorsat1": color_saturation1,
    "colorsat2": color_saturation2,
    "jpeg2000": jpeg2000,
    "jpeg": jpeg,
    "whitenoise": white_noise,
    "whitenoiseCC": white_noise_cc,
    "impulsenoise": impulse_noise,
    "multnoise": multiplicative_noise,
    "brighten": brighten,
    "darken": darken,
    "meanshift": mean_shift,
    "jitter": jitter,
    "noneccpatch": non_eccentricity_patch,
    "pixelate": pixelate,
    "quantization": quantization,
    "colorblock": color_block,
    "highsharpen": high_sharpen,
    "lincontrchange": linear_contrast_change,
    "nonlincontrchange": non_linear_contrast_change,
}


def distort_images(image: torch.Tensor, distort_functions: list = None, distort_values: list = None,
                   max_distortions: int = 4, num_levels: int = 5) -> torch.Tensor:
    """
    Distorts an image using the distortion composition obtained with the image degradation model proposed in the paper
    https://arxiv.org/abs/2310.14918.

    Args:
        image (Tensor): image to distort
        distort_functions (list): list of the distortion functions to apply to the image. If None, the functions are randomly chosen.
        distort_values (list): list of the values of the distortion functions to apply to the image. If None, the values are randomly chosen.
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        image (Tensor): distorted image
        distort_functions (list): list of the distortion functions applied to the image
        distort_values (list): list of the values of the distortion functions applied to the image
    """
    if distort_functions is None or distort_values is None:
        distort_functions, distort_values = get_distortions_composition(max_distortions, num_levels)

    for distortion, value in zip(distort_functions, distort_values):
        image = distortion(image, value)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 1)

    return image, distort_functions, distort_values


def get_distortions_composition(max_distortions: int = 7, num_levels: int = 5) -> (List[Callable], List[Union[int, float]]):
    """
    Image Degradation model proposed in the paper https://arxiv.org/abs/2310.14918. Returns a randomly assembled ordered
    sequence of distortion functions and their values.

    Args:
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        distort_functions (list): list of the distortion functions to apply to the image
        distort_values (list): list of the values of the distortion functions to apply to the image
    """
    MEAN = 0
    STD = 2.5

    num_distortions = random.randint(1, max_distortions)
    groups = random.sample(list(distortion_groups.keys()), num_distortions)
    distortions = [random.choice(distortion_groups[group]) for group in groups]
    distort_functions = [distortion_functions[dist] for dist in distortions]

    probabilities = [1 / (STD * np.sqrt(2 * np.pi)) * np.exp(-((i - MEAN) ** 2) / (2 * STD ** 2))
                     for i in range(num_levels)]  # probabilities according to a gaussian distribution
    normalized_probabilities = [prob / sum(probabilities)
                                for prob in probabilities]  # normalize probabilities
    distort_values = [np.random.choice(distortion_range[dist][:num_levels], p=normalized_probabilities) for dist
                      in distortions]

    return distort_functions, distort_values


def resize_crop(img: PILImage, crop_size: int = 224, downscale_factor: int = 1) -> PILImage:
    """
    Resize the image with the desired downscale factor and optionally crop it to the desired size. The crop is randomly
    sampled from the image. If crop_size is None, no crop is applied. If the crop is out of bounds, the image is
    automatically padded with zeros.

    Args:
        img (PIL Image): image to resize and crop
        crop_size (int): size of the crop. If None, no crop is applied
        downscale_factor (int): downscale factor to apply to the image

    Returns:
        img (PIL Image): resized and/or cropped image
    """
    w, h = img.size
    if downscale_factor > 1:
        img = img.resize((w // downscale_factor, h // downscale_factor))
        w, h = img.size

    if crop_size is not None:
        top = randrange(0, max(1, h - crop_size))
        left = randrange(0, max(1, w - crop_size))
        img = TF.crop(img, top, left, crop_size, crop_size)     # Automatically pad with zeros if the crop is out of bounds

    return img


def center_corners_crop(img: PILImage, crop_size: int = 224) -> List[PILImage]:
    """
    Return the center crop and the four corners of the image.

    Args:
        img (PIL.Image): image to crop
        crop_size (int): size of each crop

    Returns:
        crops (List[PIL.Image]): list of the five crops
    """
    width, height = img.size

    # Calculate the coordinates for the center crop and the four corners
    cx = width // 2
    cy = height // 2
    crops = [
        TF.crop(img, cy - crop_size // 2, cx - crop_size // 2, crop_size, crop_size),  # Center
        TF.crop(img, 0, 0, crop_size, crop_size),  # Top-left corner
        TF.crop(img, height - crop_size, 0, crop_size, crop_size),  # Bottom-left corner
        TF.crop(img, 0, width - crop_size, crop_size, crop_size),  # Top-right corner
        TF.crop(img, height - crop_size, width - crop_size, crop_size, crop_size)  # Bottom-right corner
    ]

    return crops
