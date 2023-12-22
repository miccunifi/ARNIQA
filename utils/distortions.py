import torch
from torchvision.io.image import decode_jpeg, encode_jpeg
from torchvision import transforms
import numpy as np
import random
import math
from torch.nn import functional as F
import io
from PIL import Image
import ctypes
import kornia

from utils.utils import PROJECT_ROOT
from utils.utils_distortions import fspecial, filter2D, curves, imscatter, mapmm

dither_cpp = ctypes.CDLL(PROJECT_ROOT / "utils" / "dither_extension/dither.so").dither
dither_cpp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                       ctypes.c_int]


def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def motion_blur(x: torch.Tensor, radius: int, angle: bool = None) -> torch.Tensor:
    if angle is None:
        angle = random.randint(0, 180)
    h = fspecial('motion', radius, angle)
    h = torch.from_numpy(h.copy()).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def color_diffusion(x: torch.Tensor, amount: int) -> torch.Tensor:
    blur_sigma = 1.5 * amount + 2
    scaling = amount
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)

    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    diff_ab = filter2D(lab[:, 1:3, ...], h.unsqueeze(0))
    lab[:, 1:3, ...] = diff_ab * scaling

    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255.) / 255.
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def color_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    def perc(x, perc):
        xs = torch.sort(x)
        i = len(xs) * perc / 100.
        i = max(min(i, len(xs)), 1)
        v = xs[round(i - 1)]
        return v

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5

    fs = 2 * math.ceil(2 * 4) + 1
    h = fspecial('gaussian', (fs, fs), 4)
    h = torch.from_numpy(h).float()

    e = filter2D(e, h.unsqueeze(0))

    mine = torch.min(e)
    maxe = torch.max(e)

    if mine < maxe:
        e = (e - mine) / (maxe - mine)

    percdev = [1, 1]
    valuehi = perc(e, 100 - percdev[1])
    valuelo = 1 - perc(1 - e, 100 - percdev[0])

    e = torch.max(torch.min(e, valuehi), valuelo)

    channel = 1
    g = x[channel, :, :]
    a = np.random.random((1, 2))
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * amount)[0].astype(int)

    y = F.pad(g, (amount_shift[0], amount_shift[0]), mode='replicate')
    y = F.pad(y.transpose(1, 0), (amount_shift[1], amount_shift[1]), mode='replicate').transpose(1, 0)
    y = torch.roll(y, (amount_shift[0], amount_shift[1]), dims=(0, 1))

    if amount_shift[1] != 0:
        y = y[amount_shift[1]:-amount_shift[1], ...]
    if amount_shift[0] != 0:
        y = y[..., amount_shift[0]:-amount_shift[0]]

    yblend = y * e + x[channel, ...] * (1 - e)
    x[channel, ...] = yblend

    return x


def color_saturation1(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    y = kornia.color.hsv_to_rgb(hsv)
    return y[[2, 1, 0], ...]


def color_saturation2(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    lab[1:3, ...] = lab[1:3, ...] * factor
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255) / 255.
    return y[[2, 1, 0], ...]


def jpeg2000(x: torch.Tensor, ratio: int) -> torch.Tensor:
    ratio = int(ratio)
    compression_params = {
        'quality_mode': 'rates',
        'quality_layers': [ratio],  # Compression ratio
        'num_resolutions': 8,  # Number of wavelet decompositions
        'prog_order': 'LRCP',  # Progression order: Layer-Resolution-Component-Position
    }

    # Compress the image and save it using the JPEG2000 format
    x *= 255.
    x = x.byte().cpu().numpy()

    x = Image.fromarray(x.transpose(1, 2, 0), 'RGB')

    with io.BytesIO() as output:
        x.save(output, format='JPEG2000', **compression_params)
        compressed_data = output.getvalue()

    y = Image.open(io.BytesIO(compressed_data))
    y = transforms.ToTensor()(y)

    return y


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    x *= 255.
    y = encode_jpeg(x.byte().cpu(), quality=quality)
    y = (decode_jpeg(y) / 255.).to(torch.float32)
    return y


def white_noise(x: torch.Tensor, var: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)

    y = x + noise

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.
    return y


def white_noise_cc(x: torch.Tensor, var: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)

    ycbcr = kornia.color.rgb_to_ycbcr(x)
    y = ycbcr + noise

    y = kornia.color.ycbcr_to_rgb(y)

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.

    return y


def impulse_noise(x: torch.Tensor, d: float, s_vs_p: float = 0.5) -> torch.Tensor:
    num_sp = int(d * x.shape[0] * x.shape[1] * x.shape[2])

    coords = np.concatenate((np.random.randint(0, 3, (num_sp, 1)),
                             np.random.randint(0, x.shape[1], (num_sp, 1)),
                             np.random.randint(0, x.shape[2], (num_sp, 1))), 1)

    num_salt = int(s_vs_p * num_sp)

    coords_salt = coords[:num_salt].transpose(1, 0)
    coords_pepper = coords[num_salt:].transpose(1, 0)

    x[coords_salt] = 1
    x[coords_pepper] = 0

    return x


def multiplicative_noise(x: torch.Tensor, var: float) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)
    y = x + x * noise
    y = torch.clip(y, 0, 1)
    return y


def brighten(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)

    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 + amount / 2)
    lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 + amount / 2)

    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)

    y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def darken(x: torch.Tensor, amount: float, dolab: bool = False) -> torch.Tensor:
    x = x[[2, 1, 0], :, :]
    lab = kornia.color.rgb_to_lab(x)
    if dolab:
        l = lab[0, ...] / 100.
        l_ = curves(l, 0.5 + amount / 2)
        lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 - amount / 2)

    if dolab:
        j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
        y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def mean_shift(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0], :, :]

    y = torch.clamp(x + amount, 0, 1)
    return y[[2, 1, 0]]


def jitter(x: torch.Tensor, amount: float) -> torch.Tensor:
    y = imscatter(x, amount, 5)
    return y


def non_eccentricity_patch(x: torch.Tensor, pnum: int) -> torch.Tensor:
    y = x
    patch_size = [16, 16]
    radius = 16
    h_min = radius
    w_min = radius
    c, h, w = x.shape

    h_max = h - patch_size[0] - radius
    w_max = w - patch_size[1] - radius

    for i in range(pnum):
        w_start = round(random.random() * (w_max - w_min)) + w_min
        h_start = round(random.random() * (h_max - h_min)) + h_min
        patch = y[:, h_start:h_start + patch_size[0], w_start:w_start + patch_size[0]]

        rand_w_start = round((random.random() - 0.5) * radius + w_start)
        rand_h_start = round((random.random() - 0.5) * radius + h_start)
        y[:, rand_h_start:rand_h_start + patch_size[0], rand_w_start:rand_w_start + patch_size[0]] = patch

    return y


def pixelate(x: torch.Tensor, strength: float) -> torch.Tensor:
    z = 0.95 - strength ** 0.6
    c, h, w = x.shape

    ylo = kornia.geometry.transform.resize(x, (int(h * z), int(w * z)), 'nearest')
    y = kornia.geometry.transform.resize(ylo, (h, w), 'nearest')

    return y


def quantization(x: torch.Tensor, levels: int) -> torch.Tensor:
    image = kornia.color.rgb_to_grayscale(x) * 255
    image = image.cpu().numpy()
    num_classes = levels

    # minimum variance thresholding
    hist, bins = np.histogram(image, num_classes, [0, 255])

    return_thresholds = np.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        return_thresholds[i] = bins[i + 1]

    # quantize image with thresholds
    bins = torch.tensor([0] + return_thresholds.tolist() + [256])
    bins = bins.type(torch.int)
    image = torch.bucketize(x.contiguous() * 255., bins).to(torch.float32)
    image = mapmm(image)
    return image


def color_block(x: torch.Tensor, pnum: int) -> torch.Tensor:
    patch_size = [32, 32]

    c, w, h = x.shape

    y = x

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    for i in range(pnum):
        color = np.random.random(3)
        px = math.floor(random.random() * w_max)
        py = math.floor(random.random() * h_max)
        patch = torch.ones((3, patch_size[0], patch_size[1]))
        for j in range(3):
            patch[j, ...] *= color[j]
        y[:, px:px + patch_size[0], py:py + patch_size[1]] = patch

    return y


def high_sharpen(x: torch.Tensor, amount: int, radius: int = 3) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    l = lab[0:1, ...].unsqueeze(0)

    filt_radius = math.ceil(radius * 2)
    fs = 2 * filt_radius + 1
    h = fspecial('gaussian', (fs, fs), filt_radius)
    h = torch.from_numpy(h).float()

    sharp_filter = torch.zeros((fs, fs))
    sharp_filter[filt_radius, filt_radius] = 1
    sharp_filter = sharp_filter - h

    sharp_filter *= amount
    sharp_filter[filt_radius, filt_radius] += 1

    l = filter2D(l, sharp_filter.unsqueeze(0))

    lab[0, ...] = l

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    y = kornia.color.lab_to_rgb(lab)
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def linear_contrast_change(x: torch.Tensor, amount: float) -> torch.Tensor:
    y = curves(x, [0.25 - amount / 4, 0.75 + amount / 4])
    return y


def non_linear_contrast_change(x: torch.Tensor, output_offset_value: float, output_central_value: float = 0.5,
                               input_offset_value: float = 0.5, input_central_value: float = 0.5) -> torch.Tensor:
    low_in = input_central_value - input_offset_value
    high_in = input_central_value + input_offset_value
    low_out = output_central_value - output_offset_value
    high_out = output_central_value + output_offset_value

    # Clip the input image to the specified input range
    x = np.clip(x, low_in, high_in)

    # Calculate the slope and intercept of the linear transformation
    slope = (high_out - low_out) / (high_in - low_in)
    intercept = low_out - slope * low_in

    # Apply the linear transformation to adjust the pixel values
    y = slope * x + intercept

    # Clip the adjusted image to the specified output range
    y = np.clip(y, low_out, high_out)

    return y
