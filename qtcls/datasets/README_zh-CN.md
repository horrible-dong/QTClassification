# 如何注册你的数据集

> 作者: QIU Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

1. 创建 `your_dataset.py`。
2. 在 `your_dataset.py` 中，继承 `BaseDataset` 来定义你的数据集。

```python
# your_dataset.py

__all__ = ['YourDataset']

from ._base_ import BaseDataset


class YourDataset(BaseDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, batch_transform=None, loader=None):
        super().__init__(root, split, transform, target_transform, batch_transform, loader)
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

    def collate_fn(self, batch):
        ...
```

3. 在 [`__init__.py`](__init__.py) 中，
    - 导入你的数据集。
    - 添加你的数据集的类别数（`num_classes`，必需）与默认目标图像尺寸（`image_size`
      ，非必需，优先级：`--image_size` > `_image_size[dataset_name]`
      ）。（数据预处理时，图像会自动缩放至设定的目标图像尺寸 `image_size`。）
    - 在 `build_dataset()`
      中注册你的数据集及其对应的数据增强方式（augmentation / transform）。我们推荐的数据增强定义格式为
      `{'train': Optional[Callable], 'val': Optional[Callable], ...}`，其中键值 `'train'`, `'val'`, `...`
      与参数 `split` 对应。例如，当创建训练集时，`split` 为 `'train'` ，于是 `transform[split]` 就得到了训练集的数据增强。

```python
# __init__.py

...

from .your_dataset import YourDataset

_num_classes = {  # 必需
    # 数据集名称须全部小写
    ...,
    'your_dataset': num_classes
}

_image_size = {  # 非必需（优先级：`--image_size` > `_image_size[dataset_name]`）
    # 数据集名称须全部小写
    ...,
    'your_dataset': image_size
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
        ...

        transform = {
            'train': ...,
            'val': ...
        }

        batch_transform = {
            'train': ...,
            'val': ...
        }

        ...

        return YourDataset(root=dataset_path,
                           split=split,
                           transform=transform,  # 也可以显式地写成 `transform=transform[split]`
                           batch_transform=batch_transform)  # 也可以显式地写成 `batch_transform=batch_transform[split]`
    ...
```

4. 当使用你的数据集时，把 `--dataset` / `-d` 赋值为你的数据集名称 `your_dataset`。注意 `your_dataset`
   不需要和你的数据集类名 `YourDataset` 保持一致。
5. 把你的数据集放在 `--data_root` 目录下 (默认是 `./data` 目录下)。
   请参考 [“如何放置你的数据集”](../../data/README_zh-CN.md) 。 
