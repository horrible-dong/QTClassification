# 如何注册你的数据集

[English](README.md) | 简体中文

1. 创建 `your_dataset.py`。
2. 在 `your_dataset.py` 中，继承 `BaseDataset` 来定义你的数据集。

```python
# your_dataset.py

from ._base_ import BaseDataset

__all__ = ['YourDataset']


class YourDataset(BaseDataset):
    def __init__(self, root, split='train', loader=None, transform=None, target_transform=None):
        super().__init__(root, split, loader, transform, target_transform)
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

    @staticmethod
    def collate_fn(batch):
        ...
```

3. 在 [`__init__.py`](__init__.py) 中，
    - 导入你的数据集。
    - 添加你的数据集的类别数（num_classes）。
    - 在 `build_dataset()`
      中注册你的数据集及其对应的数据增强方式（transform）。注意数据增强应该是 `{'train': Callable, 'val': Callable, ...}`
      格式，其中键值 `'train'`, `'val'`, `...` 与参数 `split` 对应。例如，当创建训练集时，`split` 为 `'train'`
      ，于是 `transform[split]` 就得到了训练集的数据增强。

```python
# __init__.py

...

from .your_dataset import YourDataset

num_classes = {
    # 全部小写 !!!
    ...
    'your_dataset': your_num_classes
}

def build_dataset(args, split, download=True):
    """
    split: 'train', 'val', 'test' or others
    """
    split = split.lower()
    dataset_name = args.dataset.lower()
    dataset_path = os.path.join(args.data_root, dataset_name)

    ...

    if dataset_name == 'your_dataset':
        transform = {
            "train": ...,
            "val": ...
        }

        return YourDataset(root=dataset_path,
                           split=split,
                           transform=transform)

    ...
```

4. 当使用你的数据集时，把 `--dataset` 赋值为你的数据集名称 `your_dataset`。注意 `your_dataset`
   不需要和你的数据集类名 `YourDataset` 保持一致。
5. 把你的数据集放在 `--data_root` 目录下 (默认是 `./data` 目录下).
   请参考[“如何放置你的数据集”](../../data/README_zh-CN.md)。 