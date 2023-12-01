# How to register your datasets

> Author: QIU Tian  
> Affiliate: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

1. Create `your_dataset.py`.
2. In `your_dataset.py`, inherit `BaseDataset` to define your criterion.

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

3. In [`__init__.py`](__init__.py),
    - Import your dataset.
    - Add your dataset's `num_classes` (required) and target `image_size` (optional,
      priority: `--image_size` > `_image_size[dataset_name]`). (During data preprocessing, images will be automatically
      scaled to the target `image size`.)
    - Register your dataset and its data transform (augmentation) in `build_dataset()`. We recommend that the transform
      be in `{'train': Optional[Callable], 'val': Optional[Callable], ...}` format, where the keys
      `'train'`, `'val'`, `...` correspond to the argument `split`. For example, when building the training dataset,
      `split` is set to `'train'`, and then `transform[split]` obtains the training data transform.

```python
# __init__.py

...

from .your_dataset import YourDataset

_num_classes = {  # Required
    # Dataset names must be all in lowercase.
    ...,
    'your_dataset': num_classes
}

_image_size = {  # Optional (Priority: `--image_size` > `_image_size[dataset_name]`)
    # Dataset names must be all in lowercase.
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
                           transform=transform,  # Can also be written explicitly as 'transform=transform[split]'.
                           batch_transform=batch_transform)  # Can also be written explicitly as 'batch_transform=batch_transform[split]'.
    ...
```

4. When using your dataset, set `--dataset` / `-d` to `your_dataset`. Note that `your_dataset` does not have to be
   consistent with the dataset class name `YourDataset`.
5. Put your dataset into the `--data_root` directory (default is `./data`). Please
   follow [this instruction](../../data/README.md). 
