# 如何注册你的损失函数

> 作者: QIU, Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

1. 创建 `your_criterion.py`。
2. 在 `your_criterion.py` 中，继承 `BaseCriterion` 来定义你的损失函数。**你只需要定义每一个子损失函数，它们均以字典形式返回，
   每一个键值对代表一个损失项。子损失函数的名称 \*\*必须\*\* 以 `loss_` 开头，即`loss_{name}`形式。** `forward()`
   函数定义在父类 `BaseCriterion` 中，它会调用我们定义的子损失函数，并汇总它们的结果。

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

3. 在 [`__init__.py`](__init__.py) 中，
    - 导入你的损失函数。
    - 在 `build_criterion()` 中注册你的损失函数。**在这里，`{name1,name2,name3}` 应该和 `YourCriterion`
      中定义的子损失函数的名称保持一致，`weight_dict` 中的 `{a,b,c,d}` 应该和子损失函数输出的各损失项的键值对应。**

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

4. 当使用你的损失函数时，把 `--criterion` 赋值为你的损失函数名称 `your_criterion`。注意 `your_criterion`
   不需要和你的损失函数类名 `YourCriterion` 保持一致。

我们在 [`_demo_.py`](_demo_.py) 中提供了一个损失函数样例。

```python
# _demo_.py
# Copyright (c) QIU, Tian. All rights reserved.

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
```
