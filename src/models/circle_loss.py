#  modified from https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py 

from typing import Tuple

import torch
from torch import nn, Tensor


def measure_similarity(anchor: Tensor, positive: Tensor, negative: Tensor) -> Tuple[Tensor, Tensor]:
    # sp: within-class similarity score
    # sn: between class similarity score
    sp = nn.functional.cosine_similarity(anchor, positive, dim=1)
    sn = nn.functional.cosine_similarity(anchor, negative, dim=1)

    return sp, sn


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m   # negative margin
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.) # positive margin = 1 + negative margin
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
