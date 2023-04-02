# 如何注册你的验证器

[English](README.md) | 简体中文

1. 创建 `your_evaluator.py`。
2. 在 `your_evaluator.py` 中定义你的验证器。你可以自由发挥，或者参考 [`default.py`](default.py)
   中我们的默认验证器的写法，它可以轻松计算准确率、召回率、精确率和f1分数。

```python
# your_evaluator.py

...

__all__ = ['YourEvaluator']


class YourEvaluator(...):
    ...
```

3. 在 [`__init__.py`](__init__.py) 中，

    - 导入你的验证器。

    - 在 `build_evaluator()` 中注册你的验证器。

```python
# __init__.py

...

from .your_evaluator import YourEvaluator


def build_evaluator(args):
    evaluator_name = args.evaluator.lower()

    ...

    if evaluator_name == 'your_evaluator':
        return YourEvaluator(...)

    ...
```

4. 当使用你的数据集时，把 `--evaluator` 赋值为你的验证器名称 `your_evaluator`。注意 `your_evaluator`
   不需要和你的验证器类名 `YourEvaluator` 保持一致。