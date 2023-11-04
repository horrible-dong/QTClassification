# How to register your schedulers

> Author: QIU Tian  
> Affiliate: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

1. Create `your_scheduler.py`.
2. In `your_scheduler.py`, define your scheduler.

```python
# your_scheduler.py

__all__ = ['YourScheduler']

...


class YourScheduler(...):
    ...
```

3. In [`__init__.py`](__init__.py),

    - Import your scheduler.

    - Register your scheduler in `build_scheduler()`.

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

4. When using your scheduler, set `--scheduler` to `your_scheduler`. Note that `your_scheduler` does not have to be
   consistent with the scheduler class name `YourScheduler`.
