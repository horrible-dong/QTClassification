# 如何注册你的学习率调整策略

> 作者: QIU, Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

1. 创建 `your_scheduler.py`。
2. 在 `your_scheduler.py` 中，定义你的学习率调整策略。

```python
# your_scheduler.py

...

__all__ = ['YourScheduler']


class YourScheduler(...):
    ...
```

3. 在 [`__init__.py`](__init__.py) 中，

    - 导入你的学习率调整策略。

    - 在 `build_scheduler()` 中注册你的学习率调整策略。

```python
# __init__.py

...

from .your_scheduler import YourScheduler


def build_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()

    ...

    if scheduler_name == 'your_scheduler':
        return YourScheduler(optimizer, ...)

    ...
```

4. 当使用你的学习率调整策略时，把 `--scheduler` 赋值为你的学习率调整策略名称 `your_scheduler`。注意 `your_scheduler`
   不需要和你的学习率调整策略类名 `YourScheduler` 保持一致。