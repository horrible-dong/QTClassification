# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['DemoCriterion']

from typing import List, Dict

import torch
import torch.nn.functional as F

from qtcls.criterions._base_ import BaseCriterion
from qtcls.utils.misc import accuracy


class DemoCriterion(BaseCriterion):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__(losses, weight_dict)

    def loss_class(self, outputs, targets, **kwargs):
        loss_ce = F.cross_entropy(outputs['logits'], targets['labels'], reduction='mean')
        losses = {'loss_ce': loss_ce}
        losses['class_error'] = 100 - accuracy(outputs['logits'], targets['labels'])[0]
        return losses

    def loss_bbox(self, outputs, targets, **kwargs):
        loss_boxes = F.l1_loss(outputs['boxes'], targets['boxes'], reduction='mean')
        losses = {'loss_boxes': loss_boxes}
        return losses


if __name__ == '__main__':
    torch.manual_seed(42)

    criterion = DemoCriterion(losses=['class', 'bbox'], weight_dict={'loss_ce': 1, 'loss_boxes': 2})

    outputs = {'logits': torch.randn([3, 10]), 'boxes': torch.randn([3, 4])}
    targets = {'labels': torch.tensor([2, 9, 6]), 'boxes': torch.randn([3, 4])}

    loss = criterion(outputs, targets)

    print(loss)  # {'loss_ce': tensor(2.1619), 'class_error': tensor(66.6667), 'loss_boxes': tensor(1.1626)}

    weight_dict = criterion.weight_dict
    loss_scaled = {k: v * weight_dict[k] for k, v in loss.items() if k in weight_dict}

    print(loss_scaled)  # {'loss_ce': tensor(2.1619), 'loss_boxes': tensor(2.3253)}
