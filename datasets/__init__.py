# Copyright (c) QIU, Tian. All rights reserved.

import os

from torchvision import transforms as tfs

from .cifar import CIFAR10, CIFAR100
from .folder import ImageFolder
from .imagenet import ImageNet
from .mnist import MNIST

num_classes = {
    # all in lowercase !!!
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet1k': 1000,
    'stl10': 10,
    'svhn': 10,
    'pets': 37,
}


def build_dataset(args, split, download=True):
    """
    mode: 'train', 'val', 'test' or others
    """
    split = split.lower()
    dataset_name = args.dataset.lower()
    dataset_path = os.path.join(args.data_root, dataset_name)

    if dataset_name == 'mnist':
        if split == 'val':
            split = 'test'

        transform = {
            "train": tfs.Compose([
                tfs.Resize(224),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ]),
            "test": tfs.Compose([
                tfs.Resize(224),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ])
        }

        return MNIST(root=dataset_path,
                     split=split,
                     transform=transform,
                     download=download)

    if dataset_name == 'cifar10':
        if split == 'val':
            split = 'test'

        transform = {
            "train": tfs.Compose([
                tfs.Resize(224),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize(224),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

        return CIFAR10(root=dataset_path,
                       split=split,
                       transform=transform,
                       download=download)

    if dataset_name == 'cifar100':
        if split == 'val':
            split = 'test'

        transform = {
            "train": tfs.Compose([
                tfs.Resize(224),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize(224),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

        return CIFAR100(root=dataset_path,
                        split=split,
                        transform=transform,
                        download=download)

    if dataset_name == 'imagenet1k':
        transform = {
            "train": tfs.Compose([
                tfs.Resize(256),
                tfs.CenterCrop(224),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": tfs.Compose([
                tfs.Resize(256),
                tfs.CenterCrop(224),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        return ImageNet(root=dataset_path,
                        split=split,
                        transform=transform)

    if dataset_name == 'stl10':
        raise NotImplementedError

    if dataset_name == 'pets':
        raise NotImplementedError

    if dataset_name == 'svhn':
        if split == 'val':
            split = 'test'

        transform = {
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

        return ImageFolder(root=dataset_path,
                           split=split,
                           transform=transform)

    raise ValueError(f"dataset '{dataset_name}' is not found.")
