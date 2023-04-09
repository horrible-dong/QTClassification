# How to register your datasets

English | [简体中文](README_zh-CN.md)

1. Create `your_dataset.py`.
2. In `your_dataset.py`, inherit `BaseDataset` to define your criterion.

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

3. In [`__init__.py`](__init__.py),
    - Import your dataset.
    - Add your dataset's num_classes.
    - Register your dataset and its data transform in `build_dataset()`. Note that the transform should be
      in `{'train': Callable, 'val': Callable, ...}` format, where the keys `'train'`, `'val'`, `...` correspond to the
      argument `split`. For example, when building the training dataset, `split` is set to `'train'`, and
      then `transform[split]` obtains the training data transform.

```python
# __init__.py

...

from .your_dataset import YourDataset

num_classes = {
    # all in lowercase !!!
    ...,
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

4. When using your dataset, set `--dataset` to `your_dataset`. Note that `your_dataset` does not have to be consistent
   with the dataset class name `YourDataset`.
5. Put your dataset into the `--data_root` directory (default is `./data`). Please
   follow [this instruction](../../data/README.md). 