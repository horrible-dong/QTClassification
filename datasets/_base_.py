# Copyright (c) QIU, Tian. All rights reserved.

import os.path

from torch.utils.data import Dataset

from utils.io import pil_loader

__all__ = ["BaseDataset"]


def default_loader(path, format="RGB"):
    return pil_loader(path, format)


class BaseDataset(Dataset):
    def __init__(self, root, split='train', loader=None, transform=None, target_transform=None, verbose=True):
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
            print(f'loading {self.__class__.__name__.lower()}-{split} from {self.root}')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError

    def __check_transforms(self):
        if isinstance(self.transform, dict):
            self.transform = self.transform[self.split]
        if isinstance(self.target_transform, dict):
            self.target_transform = self.target_transform[self.split]
