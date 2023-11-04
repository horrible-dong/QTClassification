# ---------------------------------------
# Modified from torchvision by QIU Tian
# ---------------------------------------

__all__ = ['SVHN']

import os
from typing import Tuple, Any

import numpy as np
from PIL import Image
from torchvision.datasets.utils import verify_str_arg, check_integrity, download_url

from ._base_ import BaseDataset
from ..utils.decorators import main_process_only


class SVHN(BaseDataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset."""

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(self, root, split, transform=None, target_transform=None, batch_transform=None, download=False):
        split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        super().__init__(root, split, transform, target_transform, batch_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    @main_process_only
    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

