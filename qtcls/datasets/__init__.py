# Copyright (c) QIU, Tian. All rights reserved.

from .cifar import CIFAR10, CIFAR100
from .folder import ImageFolder
from .imagenet import ImageNet
from .mnist import MNIST
from .oxford_iiit_pet import OxfordIIITPet
from .stl10 import STL10
from .svhn import SVHN

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
    split: 'train', 'val', 'test' or others
    """
    import os
    from torchvision import transforms as tfs

    split = split.lower()
    dataset_name = args.dataset.lower()
    dataset_path = os.path.join(args.data_root, dataset_name)

    if dataset_name == 'mnist':  # 28 x 28, ** 1 channel **
        if split == 'val':
            split = 'test'

        image_size = 28 if args.image_size is None else args.image_size

        transform = {
            "train": tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ]),
            "test": tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ])
        }

        return MNIST(root=dataset_path,
                     split=split,
                     transform=transform,
                     download=download)

    if dataset_name == 'cifar10':  # 32 x 32
        if split == 'val':
            split = 'test'

        image_size = 32 if args.image_size is None else args.image_size

        transform = {
            "train": tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.Resize(image_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

        return CIFAR10(root=dataset_path,
                       split=split,
                       transform=transform,
                       download=download)

    if dataset_name == 'cifar100':  # 32 x 32
        if split == 'val':
            split = 'test'

        image_size = 32 if args.image_size is None else args.image_size

        transform = {
            "train": tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.Resize(image_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize(image_size),
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
                tfs.RandomResizedCrop(224),
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

        return ImageFolder(root=dataset_path,
                           split=split,
                           transform=transform)

        # return ImageNet(root=dataset_path,
        #                 split=split,
        #                 transform=transform)

    if dataset_name == 'stl10':  # 96 x 96
        if split == 'val':
            split = 'test'

        transform = {
            "train": tfs.Compose([
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        return STL10(root=dataset_path,
                     split=split,
                     transform=transform,
                     download=download)

    if dataset_name == 'svhn':  # 32 x 32
        if split == 'val':
            split = 'test'

        image_size = 32 if args.image_size is None else args.image_size

        transform = {
            "train": tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            "test": tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

        return SVHN(root=dataset_path,
                    split=split,
                    transform=transform,
                    download=download)

    if dataset_name == 'pets':
        if split == 'train':
            split = 'trainval'
        if split == 'val':
            split = 'test'

        transform = {
            "trainval": tfs.Compose([
                tfs.RandomResizedCrop(224),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": tfs.Compose([
                tfs.Resize(256),
                tfs.CenterCrop(224),
                tfs.ToTensor(),
                tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        return OxfordIIITPet(root=dataset_path,
                             split=split,
                             transform=transform,
                             download=download)

    raise ValueError(f"dataset '{dataset_name}' is not found.")
