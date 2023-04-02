# Copyright (c) QIU, Tian. All rights reserved.

from torch import nn

__all__ = ['BaseCriterion']


class BaseCriterion(nn.Module):
    def __init__(self, losses: list, weight_dict: dict):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets))
        return losses
