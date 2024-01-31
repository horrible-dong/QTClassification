# How to register your optimizers

> Author: QIU Tian  
> Affiliation: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

1. Create `your_optimizer.py`.
2. In `your_optimizer.py`, define your optimizer.

```python
# your_optimizer.py

__all__ = ['YourOptimizer']

...


class YourOptimizer(Optimizer):
    ...
```

3. In [`__init__.py`](__init__.py),

    - Import your optimizer.

    - Register your optimizer in `build_optimizer()`.

```python
# __init__.py

...

from .your_optimizer import YourOptimizer


def build_optimizer(args, params):
    optimizer_name = args.optimizer.lower()

    ...

    if optimizer_name == 'your_optimizer':
        return YourOptimizer(params, ...)

    ...
```

4. When using your optimizer, set `--optimizer` to `your_optimizer`. Note that `your_optimizer` does not have to be
   consistent with the optimizer class name `YourOptimizer`.
