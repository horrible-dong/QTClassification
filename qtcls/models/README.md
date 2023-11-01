# How to register your models

> Author: QIU, Tian  
> Affiliate: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

1. Create `your_model.py`.
2. In `your_model.py`, inherit `nn.Module` to define your model. **Each of your model architecture returns with a
   function.**

```python
# your_model.py

# We recommend you to declear '__all__'.
__all__ = ['your_model_architecture_1', 'your_model_architecture_2', 'your_model_architecture_3']

from torch import nn


class YourModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, ...):
        ...


def your_model_architecture_1(...):
    return YourModel(...)


def your_model_architecture_2(...):
    return YourModel(...)


def your_model_architecture_3(...):
    return YourModel(...)
```

3. In [`__init__.py`](__init__.py), import all your model architectures.

```python
# __init__.py

...

from .your_dataset import *  # Usually used when '__all__' has been declared in your_model.py.

# When '__all__' has not been declared in your_model.py, use:
# from .your_dataset import your_model_architecture_1, your_model_architecture_2, your_model_architecture_3

...
```

4. If you have pretrained weights, add its path/url into [`_pretrain_.py`](_pretrain_.py). When loading pretrained
   weights, `model_local_paths` will be searched first, and then `model_urls`. For the same model architecture, you can
   store multiple local weight paths in a list, and the system will automatically search from them.

```python
# _pretrain_.py

# local paths (high priority)
model_local_paths = {
    "your_model_architecture_1": "/local/path/to/the/pretrained",
    "your_model_architecture_2": "/local/path/to/the/pretrained",
    "your_model_architecture_3": ["/local/path_1/to/the/pretrained", "/local/path_2/to/the/pretrained"],
}

# urls (low priority)
model_urls = {
    "your_model_architecture_1": "url://to/the/pretrained",
    "your_model_architecture_2": "url://to/the/pretrained",
    "your_model_architecture_3": "url://to/the/pretrained",
}
```

**Note:**
To temporarily use a pre-trained weight path, you can specify it by command-line argument `--pretrain` / `-p`. For
long-term use of a pre-trained weight path, it is preferable to write it in [`_pretrain_.py`](_pretrain_.py).
Priority: `--pretrain` > `model_local_paths` > `model_urls`.

5. When using your model, set `--model` / `-m` to `your_model_architecture_{1/2/3}`. 
