# 如何注册你的模型

> 作者: QIU, Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

1. 创建 `your_model.py`。
2. 在 `your_model.py` 中，继承 `nn.Module` 来定义你的模型。每种模型结构分别通过一个函数返回。

```python
# your_model.py

from torch import nn

# 我们推荐你申明 '__all__' 变量。
__all__ = ['your_model_architecture_1', 'your_model_architecture_2', 'your_model_architecture_3']


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

3. 在 [`__init__.py`](__init__.py) 中，导入你所有的模型架构（model architectures）。

```python
# __init__.py

...

from .your_dataset import *  # 通常在 '__all__' 被申明的情况下可以这么写。

# 当 '__all__' 在 your_model.py 中未被申明时，使用:
# from .your_dataset import your_model_architecture_1, your_model_architecture_2, your_model_architecture_3

...
```

4. 如果你有预训练权重，把它的路径添加到 [`_pretrain_.py`](_pretrain_.py)
   中。当加载预训练权重时，会优先搜索 `model_local_paths`，而后才是 `model_urls`。对于同一个模型架构，你可以用列表存放多个本地权重路径，系统会自动从中搜索。

```python
# _pretrain_.py

# 本地路径（高优先级）
model_local_paths = {
    "your_model_architecture_1": "/local/path/to/the/pretrained",
    "your_model_architecture_2": "/local/path/to/the/pretrained",
    "your_model_architecture_3": ["/local/path_1/to/the/pretrained", "/local/path_2/to/the/pretrained"],
}

# 云端路径（低优先级）
model_urls = {
    "your_model_architecture_1": "url://to/the/pretrained",
    "your_model_architecture_2": "url://to/the/pretrained",
    "your_model_architecture_3": "url://to/the/pretrained",
}
```

4. 当使用你的模型时，把 `--model` / `-m` 赋值为你的模型架构名称 `your_model_architecture_{1/2/3}`. 