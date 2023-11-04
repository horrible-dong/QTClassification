# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['BaseCriterion']

from typing import List, Dict

from torch import nn


class BaseCriterion(nn.Module):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict

    def forward(self, outputs, targets, **kwargs):
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets, **kwargs))
        return losses
