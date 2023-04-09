# Copyright (c) QIU, Tian. All rights reserved.

import os.path
from typing import Optional, Callable, Dict, Union

import torch
from torch.utils.data import Dataset

from ..utils.io import pil_loader

__all__ = ["BaseDataset"]


def default_loader(path, format="RGB"):
    return pil_loader(path, format)


class BaseDataset(Dataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            loader: Optional[Callable] = None,
            transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
            target_transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
            verbose: bool = True
    ):
        if root is None:
            root = f'./data/{self.__class__.__name__.lower()}'
        if loader is None:
            loader = default_loader
        self.root = os.path.expanduser(root)
        self.split = split.lower()
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.__check_transforms()

        if verbose:
            print(f'Loading {self.__class__.__name__.lower()}-{split} from {self.root}')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        batch_image, batch_target = list(zip(*batch))
        return torch.stack(batch_image), torch.tensor(batch_target)

    def __check_transforms(self):
        if isinstance(self.transform, dict):
            self.transform = self.transform[self.split]
        if isinstance(self.target_transform, dict):
            self.target_transform = self.target_transform[self.split]
