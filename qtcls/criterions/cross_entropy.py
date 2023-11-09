# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['CrossEntropy', 'LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy']

from typing import List, Dict

import torch
import torch.nn.functional as F

from ._base_ import BaseCriterion
from ..utils.misc import accuracy


class CrossEntropy(BaseCriterion):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__(losses, weight_dict)

    def loss_labels(self, outputs, targets, **kwargs):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'loss_labels(self, outputs, targets, **kwargs)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs["logits"]

        loss_ce = F.cross_entropy(outputs, targets, reduction='mean')
        losses = {'loss_ce': loss_ce, 'class_error': 100 - accuracy(outputs, targets)[0]}

        return losses


class LabelSmoothingCrossEntropy(BaseCriterion):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float], smoothing: float = 0.1):
        super().__init__(losses, weight_dict)
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def loss_labels(self, outputs, targets, **kwargs):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'loss_labels(self, outputs, targets, **kwargs)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs["logits"]

        if self.training:
            logprobs = F.log_softmax(outputs, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss_ce = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
        else:
            loss_ce = F.cross_entropy(outputs, targets, reduction='mean')

        losses = {'loss_ce': loss_ce, 'class_error': 100 - accuracy(outputs, targets)[0]}

        return losses


class SoftTargetCrossEntropy(BaseCriterion):  # backward compatible with 'CrossEntropy'
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__(losses, weight_dict)

    def loss_labels(self, outputs, targets, **kwargs):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'loss_labels(self, outputs, targets, **kwargs)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs["logits"]

        if targets.dim() == 1:
            loss_ce = F.cross_entropy(outputs, targets, reduction='mean')
            losses = {'loss_ce': loss_ce, 'class_error': 100 - accuracy(outputs, targets)[0]}
        else:
            loss_ce = torch.sum(-targets * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            losses = {'loss_ce': loss_ce}

        return losses
