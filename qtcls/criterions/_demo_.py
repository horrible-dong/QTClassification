# Copyright (c) QIU, Tian. All rights reserved.

import torch
import torch.nn.functional as F

from qtcls.criterions._base_ import BaseCriterion
from qtcls.utils.misc import accuracy

__all__ = ['DemoCriterion']


class DemoCriterion(BaseCriterion):
    def __init__(self, losses: list, weight_dict: dict):
        super().__init__(losses, weight_dict)

    def loss_labels(self, outputs, targets, **kwargs):
        loss_ce = F.cross_entropy(outputs['logits'], targets['logits'], reduction='mean')
        losses = {'loss_ce': loss_ce}
        losses['class_error'] = 100 - accuracy(outputs['logits'], targets['logits'])[0]
        return losses

    def loss_boxes(self, outputs, targets, **kwargs):
        loss_boxes = F.l1_loss(outputs['boxes'], targets['boxes'], reduction='mean')
        losses = {'loss_boxes': loss_boxes}
        return losses


if __name__ == '__main__':
    criterion = DemoCriterion(losses=['labels', 'boxes'],
                              weight_dict={'loss_ce': 1, 'loss_boxes': 2})

    outputs = {'logits': torch.nn.Softmax(dim=1)(torch.randn([3, 10])), 'boxes': torch.randn([3, 4])}
    targets = {'logits': torch.tensor([1, 2, 3]), 'boxes': torch.randn([3, 4])}

    loss = criterion(outputs, targets)
    print(loss)
