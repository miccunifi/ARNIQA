import torch
import torch.nn as nn
from typing import Tuple
# Avoid warning related to loading a jit model from torch.hub
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")

from models.resnet import ResNet

dependencies = ["torch"]

available_datasets = ["live", "csiq", "tid2013", "kadid10k", "flive", "spaq"]

available_datasets_ranges = {
    "live": (1, 100),
    "csiq": (0, 1),
    "tid2013": (0, 9),
    "kadid10k": (1, 5),
    "flive": (1, 100),
    "spaq": (1, 100)
}

available_datasets_mos_types = {
    "live": "dmos",
    "csiq": "dmos",
    "tid2013": "mos",
    "kadid10k": "mos",
    "flive": "mos",
    "spaq": "mos"
}

base_url = "https://github.com/miccunifi/ARNIQA/releases/download/weights"


class ARNIQA(nn.Module):
    """
    ARNIQA model for No-Reference Image Quality Assessment (NR-IQA). It is composed of a ResNet-50 encoder and a Ridge
    regressor. The regressor is trained on the dataset specified by the parameter 'regressor_dataset'. The model takes
    in input an image both at full-scale and half-scale. The output is the predicted quality score. By default, the
    predicted quality scores are in the range [0, 1], where higher is better. In addition to the score, the forward
    function allows returning the concatenated embeddings of the image at full-scale and half-scale. Also, the model can
    return the unscaled score (i.e. in the range of the training dataset).
    """
    def __init__(self, regressor_dataset: str = "kadid10k"):
        super(ARNIQA, self).__init__()
        assert regressor_dataset in available_datasets, f"parameter training_dataset must be in {available_datasets}"
        self.regressor_dataset = regressor_dataset
        self.encoder = ResNet(embedding_dim=128, pretrained=True, use_norm=True)
        self.encoder.load_state_dict(torch.hub.load_state_dict_from_url(f"{base_url}/ARNIQA.pth", progress=True,
                                                                      map_location="cpu"))
        self.encoder.eval()
        self.regressor: nn.Module = torch.hub.load_state_dict_from_url(f"{base_url}/regressor_{regressor_dataset}.pth",
                                                                        progress=True, map_location="cpu")
        self.regressor.eval()

    def forward(self, img, img_ds, return_embedding: bool = False, scale_score: bool = True):
        f, _ = self.encoder(img)
        f_ds, _ = self.encoder(img_ds)
        f_combined = torch.hstack((f, f_ds))
        score = self.regressor(f_combined)
        if scale_score:
            score = self._scale_score(score)
        if return_embedding:
            return score, f_combined
        else:
            return score

    def _scale_score(self, score: float, new_range: Tuple[float, float] = (0., 1.)) -> float:
        """
        Scale the score in the range [0, 1], where higher is better.

        Args:
            score (float): score to scale
            new_range (Tuple[float, float]): new range of the scores
        """

        # Compute scaling factors
        original_range = (available_datasets_ranges[self.regressor_dataset][0], available_datasets_ranges[self.regressor_dataset][1])
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        # Invert the scale if needed
        if available_datasets_mos_types[self.regressor_dataset] == "dmos":
            scaled_score = new_range[1] - scaled_score

        return scaled_score
