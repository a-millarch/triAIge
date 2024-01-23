import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import  Module
import torch.nn.functional as F 
from typing import Optional

class FocalLoss(Module):
    """ Weighted, multiclass focal loss"""

    def __init__(self, alpha:Optional[Tensor]=None, gamma:float=2., reduction:str='mean'):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper. Defaults to 2.
            reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        """
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none')

    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        log_p = F.log_softmax(x, dim=-1)
        pt = log_p[torch.arange(len(x)), y].exp()
        ce = self.nll_loss(log_p, y)
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

