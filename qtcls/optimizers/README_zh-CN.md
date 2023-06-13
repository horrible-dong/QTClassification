# 如何注册你的优化器

> 作者: QIU, Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

1. 创建 `your_optimizer.py`。
2. 在 `your_optimizer.py` 中，定义你的优化器。

```python
# your_optimizer.py

...

__all__ = ['YourOptimizer']


class YourOptimizer(Optimizer):
    ...
```

3. 在 [`__init__.py`](__init__.py) 中，

    - 导入你的优化器。

    - 在 `build_optimizer()` 中注册你的优化器。

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

4. 当使用你的优化器时，把 `--optimizer` 赋值为你的优化器名称 `your_optimizer`。注意 `your_optimizer`
   不需要和你的优化器类名 `YourOptimizer` 保持一致。