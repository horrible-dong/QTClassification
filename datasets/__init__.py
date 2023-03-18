# Copyright (c) QIU, Tian. All rights reserved.

import os

from torchvision import transforms as tfs

from .cifar import CIFAR10, CIFAR100
from .mnist import MNIST


def build_dataset(args, mode, download=True):
    """
    mode: 'train', 'val', 'test' or others
    """
    mode = mode.lower()
    dataset_name = args.dataset.lower()
    dataset_path = os.path.join(args.data_root, dataset_name)

    if dataset_name == 'mnist':
        if mode == 'val':
            mode = 'test'

        transforms = {
            "train": tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ]),
            "test": tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ])
        }

        return MNIST(root=dataset_path,
                     mode=mode,
                     transforms=transforms,
                     download=download)

    if dataset_name == 'cifar10':
        if mode == 'val':
            mode = 'test'

        transforms = {
            "train": tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

        return CIFAR10(root=dataset_path,
                       mode=mode,
                       transforms=transforms,
                       download=download)

    if dataset_name == 'cifar100':
        if mode == 'val':
            mode = 'test'

        transforms = {
            "train": tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

        return CIFAR100(root=dataset_path,
                        mode=mode,
                        transforms=transforms,
                        download=download)

    raise ValueError(f'{dataset_name} is not exist.')
