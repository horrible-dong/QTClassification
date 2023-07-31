# ---------------------------------------
# Modified from torchvision by QIU, Tian
# ---------------------------------------

__all__ = ['Food101']

import json
from pathlib import Path
from typing import Any, Tuple

from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive

from ._base_ import BaseDataset
from ..utils.decorators import main_process_only


class Food101(BaseDataset):
    """`Food-101 <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

        The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
        For each class, 250 manually reviewed test images are provided as well as 750 training images.
        On purpose, the training images were not cleaned, and thus still contain some amount of noise.
        This comes mostly in the form of intense colors and sometimes wrong labels. All images were
        rescaled to have a maximum side length of 512 pixels.
    """

    _URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(self, root, split, transform=None, target_transform=None, batch_transform=None, loader=None,
                 download=False):
        split = verify_str_arg(split, "split", ("train", "test"))

        super().__init__(root, split, transform, target_transform, batch_transform, loader)

        self._base_folder = Path(self.root)
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = self.loader(image_file, format="RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self.split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    @main_process_only
    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)

        import os
        import shutil

        src_dir = self._base_folder / "food-101"
        dst_dir = self._base_folder

        for filename in os.listdir(src_dir):
            src_file = src_dir / str(filename)
            dst_file = dst_dir / str(filename)
            shutil.move(src_file, dst_file)

        shutil.rmtree(src_dir)
