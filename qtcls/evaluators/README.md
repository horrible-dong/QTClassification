# How to register your evaluators

> Author: QIU Tian  
> Affiliate: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

1. Create `your_evaluator.py`.
2. In `your_evaluator.py`, define your evaluator. You are free to play, or you can refer to our default evaluator that
   computes the accuracy, recall, precision, and f1-score in [`default.py`](default.py).

```python
# your_evaluator.py

__all__ = ['YourEvaluator']

...


class YourEvaluator(...):
    ...
```

3. In [`__init__.py`](__init__.py),

    - Import your evaluator.

    - Register your evaluator in `build_evaluator()`.

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

4. When using your evaluator, set `--evaluator` to `your_evaluator`. Note that `your_evaluator` does not have to be
   consistent with the evaluator class name `YourEvaluator`.
