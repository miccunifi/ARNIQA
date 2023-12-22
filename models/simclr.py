import torch
import torch.nn
from dotmap import DotMap

from models.resnet import ResNet


class SimCLR(torch.nn.Module):
    """
    SimCLR model class used for pre-training the encoder for IQA.

    Args:
        encoder_params (dict): encoder parameters with keys
            - embedding_dim (int): embedding dimension of the encoder projection head
            - pretrained (bool): whether to use pretrained weights for the encoder
            - use_norm (bool): whether normalize the embeddings
        temperature (float): temperature for the loss function. Default: 0.1

    Returns:
        if training:
            loss (torch.Tensor): loss value
        if not training:
            q (torch.Tensor): image embeddings before the projection head (NxC)
            proj_q (torch.Tensor): image embeddings after the projection head (NxC)

    """

    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super().__init__()

        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)

        self.temperature = temperature
        self.criterion = nt_xent_loss

    def forward(self, im_q, im_k=None):
        q, proj_q = self.encoder(im_q)

        if not self.training:
            return q, proj_q

        k, proj_k = self.encoder(im_k)
        loss = self.criterion(proj_q, proj_k, self.temperature)
        return loss


def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1):
    """
    Compute the NT-Xent loss.

    Args:
        a (torch.Tensor): first set of features
        b (torch.Tensor): second set of features
        tau (float): temperature parameter
    """
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)
