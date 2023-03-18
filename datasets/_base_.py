# Copyright (c) QIU, Tian. All rights reserved.

import os.path

from torch.utils.data import Dataset

from utils.io import pil_loader

__all__ = ["BaseDataset"]


def default_loader(path, format="RGB"):
    return pil_loader(path, format)


class BaseDataset(Dataset):
    def __init__(self, root, mode='train', loader=default_loader, transforms=None, target_transforms=None,
                 verbose=True):
        if root is None:
            root = f'./data/{self.__class__.__name__.lower()}'
        self.root = os.path.expanduser(root)
        self.mode = mode.lower()
        self.loader = loader
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.__check_transforms()

        if verbose:
            print(f'loading {self.__class__.__name__.lower()}-{mode} from {self.root}')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError

    def __check_transforms(self):
        if isinstance(self.transforms, dict):
            self.transforms = self.transforms[self.mode]
        if isinstance(self.target_transforms, dict):
            self.target_transforms = self.target_transforms[self.mode]
