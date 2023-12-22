import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50


class ResNet(nn.Module):
    """
    ResNet model with a projection head.

    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): whether to normalize the embeddings
    """
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        if self.pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1   # V1 weights work better than V2
        else:
            weights = None
        self.model = resnet50(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
