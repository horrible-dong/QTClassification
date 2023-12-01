# Copyright (c) QIU Tian. All rights reserved.

from .cifar import CIFAR10, CIFAR100
from .fakedata import FakeData
from .flowers102 import Flowers102
from .food import Food101
from .imagenet import ImageNet
from .mnist import MNIST, FashionMNIST
from .oxford_iiit_pet import OxfordIIITPet
from .stanford_cars import StanfordCars
from .stl10 import STL10
from .svhn import SVHN

_num_classes = {  # Required
    # Dataset names must be all in lowercase.
    'mnist': 10,
    'fashion_mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet1k': 1000,
    'imagenet21k': 21843,
    'imagenet22k': 21843,
    'stl10': 10,
    'svhn': 10,
    'pets': 37,
    'flowers': 102,
    'cars': 196,
    'food': 101,

    'fake_data': 1000,
}

_image_size = {  # Optional (Priority: `--image_size` > `_image_size[dataset_name]`)
    # Dataset names must be all in lowercase.
    'mnist': 28,
    'fashion_mnist': 28,
    'cifar10': 32,
    'cifar100': 32,
    'imagenet1k': 224,
    'imagenet21k': 224,
    'imagenet22k': 224,
    'stl10': 96,
    'svhn': 32,
    'pets': 224,
    'flowers': 224,
    'cars': 224,
    'food': 224,

    'fake_data': 224,
}


def build_dataset(args, split, download=True):
    """
    split: 'train', 'val', 'test' or others
    """
    import os
    from torchvision import transforms as tfs
    from timm.data import create_transform, Mixup

    split = split.lower()
    dataset_name = args.dataset.lower() if not args.dummy else args.dataset.lower() + '(fakedata)'
    dataset_path = os.path.join(args.data_root, dataset_name)
    image_size = (_image_size[dataset_name] if not args.dummy
                  else _image_size[dataset_name[:dataset_name.find('(fakedata)')]]) \
        if args.image_size is None else args.image_size

    if dataset_name == 'mnist':  # ** 1 channel, set 'in_chans=1' in 'args.model_kwargs' **
        if split == 'val':
            split = 'test'

        transform = {
            'train': tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ]),
            'test': tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ])
        }

        return MNIST(root=dataset_path,
                     split=split,
                     transform=transform,
                     download=download)

    if dataset_name == 'fashion_mnist':  # ** 1 channel, set 'in_chans=1' in 'args.model_kwargs' **
        if split == 'val':
            split = 'test'

        transform = {
            'train': tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ]),
            'test': tfs.Compose([
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.5], [0.5])
            ])
        }

        return FashionMNIST(root=dataset_path,
                            split=split,
                            transform=transform,
                            download=download)

    if dataset_name == 'cifar10':
        if split == 'val':
            split = 'test'

        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return CIFAR10(root=dataset_path,
                       split=split,
                       transform=transform,
                       batch_transform=None,
                       download=download)

    if dataset_name == 'cifar100':
        if split == 'val':
            split = 'test'

        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return CIFAR100(root=dataset_path,
                        split=split,
                        transform=transform,
                        batch_transform=None,
                        download=download)

    if dataset_name in ['imagenet1k', 'imagenet21k', 'imagenet22k']:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'val': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }
        batch_transform = {
            'train': Mixup(**aug_kwargs['train_batch_aug_kwargs']),
            'val': None
        }

        return ImageNet(root=dataset_path,
                        split=split,
                        transform=transform,
                        batch_transform=batch_transform)

    if dataset_name == 'stl10':
        if split == 'val':
            split = 'test'

        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return STL10(root=dataset_path,
                     split=split,
                     transform=transform,
                     batch_transform=None,
                     download=download)

    if dataset_name == 'svhn':
        if split == 'val':
            split = 'test'

        transform = {
            'train': tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.Resize(image_size),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]),
            'test': tfs.Compose([
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

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'trainval': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return OxfordIIITPet(root=dataset_path,
                             split=split,
                             transform=transform,
                             batch_transform=None,
                             download=download)

    if dataset_name == 'flowers':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'val': create_transform(**aug_kwargs['eval_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return Flowers102(root=dataset_path,
                          split=split,
                          transform=transform,
                          batch_transform=None,
                          download=download)

    if dataset_name == 'cars':
        if split == 'val':
            split = 'test'

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return StanfordCars(root=dataset_path,
                            split=split,
                            transform=transform,
                            batch_transform=None,
                            download=download)

    if dataset_name == 'food':
        if split == 'val':
            split = 'test'

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std, _num_classes[dataset_name])
        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return Food101(root=dataset_path,
                       split=split,
                       transform=transform,
                       batch_transform=None,
                       download=download)

    if args.dummy or dataset_name == 'fake_data':
        if dataset_name != 'fake_data':
            dataset_name = dataset_name[:dataset_name.find('(fakedata)')]

        return FakeData(size=5000 if split == 'train' else 1000,
                        split=split,
                        image_size=(3, image_size, image_size),
                        num_classes=_num_classes[dataset_name],
                        transform=tfs.ToTensor(),
                        batch_transform=None)

    raise ValueError(f"Dataset '{dataset_name}' is not found.")


def build_timm_aug_kwargs(args, image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                          num_classes=1000):
    train_aug_kwargs = dict(input_size=image_size, is_training=True, use_prefetcher=False, no_aug=False,
                            scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), hflip=0.5, vflip=0., color_jitter=0.4,
                            auto_augment='rand-m9-mstd0.5-inc1', interpolation='random', mean=mean, std=std,
                            re_prob=0.25, re_mode='pixel', re_count=1, re_num_splits=0, separate=False)

    eval_aug_kwargs = dict(input_size=image_size, is_training=False, use_prefetcher=False, no_aug=False, crop_pct=0.875,
                           interpolation='bilinear', mean=mean, std=std)

    train_batch_aug_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                  mode='batch', label_smoothing=0.1, num_classes=num_classes)

    eval_batch_aug_kwargs = dict()

    train_aug_kwargs.update(args.train_aug_kwargs)
    eval_aug_kwargs.update(args.eval_aug_kwargs)
    train_batch_aug_kwargs.update(args.train_batch_aug_kwargs)
    eval_batch_aug_kwargs.update(args.eval_batch_aug_kwargs)

    return {
        'train_aug_kwargs': train_aug_kwargs,
        'eval_aug_kwargs': eval_aug_kwargs,
        'train_batch_aug_kwargs': train_batch_aug_kwargs,
        'eval_batch_aug_kwargs': eval_batch_aug_kwargs
    }
