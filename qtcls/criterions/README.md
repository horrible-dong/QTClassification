# How to register your criterions

> Author: QIU Tian  
> Affiliation: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

1. Create `your_criterion.py`.
2. In `your_criterion.py`, inherit `BaseCriterion` to define your criterion. **You only need to define every loss
   function that returns in `dict` format, and each key-value pair represents a loss item. The loss function name MUST
   be in `loss_{name}` format.** The `forward()` function is defined in the parent class `BaseCriterion` that calls the
   loss functions we defined and gathers each loss function's output.

```python
# your_criterion.py

__all__ = ['YourCriterion']

from ._base_ import BaseCriterion


class YourCriterion(BaseCriterion):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__(losses, weight_dict)
        ...

    def loss_name1(self, outputs, targets, **kwargs):
        losses = {}
        ...
        losses['a'] = ...
        return losses

    def loss_name2(self, outputs, targets, **kwargs):
        losses = {}
        ...
        losses['b'] = ...
        losses['c'] = ...
        return losses

    def loss_name3(self, outputs, targets, **kwargs):
        losses = {}
        ...
        losses['d'] = ...
        return losses
```

3. In [`__init__.py`](__init__.py),
    - Import your criterion.
    - Register your criterion in `build_criterion()`, **where `{name1,name2,name3}` should be consistent with the loss
      functions defined in `YourCriterion`, and `{a,b,c,d}` in `weight_dict` correspond to the output keys of loss
      functions.**

```python
# __init__.py

...

from .your_criterion import YourCriterion


def build_criterion(args):
    criterion_name = args.criterion.lower()

    ...

    if criterion_name == 'your_criterion':
        losses = ['name1', 'name2', 'name3'],
        weight_dict = {'a': w_a, 'b': w_b, 'c': w_c, 'd': w_d}
        return YourCriterion(losses=losses, weight_dict=weight_dict)

    ...
```

4. When using your criterion, set `--criterion` to `your_criterion`. Note that `your_criterion` does not have to be
   consistent with the criterion class name `YourCriterion`.

We've provided a demo criterion code in [`_demo_.py`](_demo_.py):

```python
# _demo_.py
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
    torch.manual_seed(42)

    criterion = DemoCriterion(losses=['labels', 'boxes'], weight_dict={'loss_ce': 1, 'loss_boxes': 2})

    outputs = {'logits': torch.nn.Softmax(dim=1)(torch.randn([3, 10])), 'boxes': torch.randn([3, 4])}
    targets = {'logits': torch.tensor([1, 2, 3]), 'boxes': torch.randn([3, 4])}

    loss = criterion(outputs, targets)

    print(loss)  # {'loss_ce': tensor(2.3039), 'class_error': tensor(100.), 'loss_boxes': tensor(1.1626)}

    weight_dict = criterion.weight_dict
    loss_scaled = {k: v * weight_dict[k] for k, v in loss.items() if k in weight_dict}

    print(loss_scaled)  # {'loss_ce': tensor(2.3039), 'loss_boxes': tensor(2.3253)}
```
