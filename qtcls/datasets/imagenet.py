# Copyright (c) QIU Tian. All rights reserved.

"""
How to download ImageNet-1K:
https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

How to make ImageNet-1K into a `folder` format:
https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
"""

__all__ = ['ImageNet']

from .folder import ImageFolder


class ImageNet(ImageFolder):
    ...
