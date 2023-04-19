# Copyright (c) QIU, Tian. All rights reserved.

import torch.nn.functional as F

from ._base_ import BaseCriterion
from ..utils.misc import accuracy


class DefaultCriterion(BaseCriterion):
    def __init__(self, losses: list, weight_dict: dict):
        super().__init__(losses, weight_dict)

    def loss_labels(self, outputs, targets, **kwargs):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'loss_labels(self, outputs, targets, **kwargs)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs["logits"]
        loss_ce = F.cross_entropy(outputs, targets, reduction='mean')
        losses = {'loss_ce': loss_ce}
        losses['class_error'] = 100 - accuracy(outputs, targets)[0]
        return losses
