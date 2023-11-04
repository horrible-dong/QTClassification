# How to put your dataset

> Author: QIU Tian  
> Affiliate: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

**The default `--data_root` is `./data`. Put your dataset in this directory.**

Each dataset is stored in a separate directory. For example,

```
data/
  ├── mnist/
  ├── cifar10/
  ├── cifar100/
  ├── imagenet1k/
  ├── imagenet21k/
  ├── stl10/
  ├── svhn/
  ├── pets/
  ├── flowers/
  ├── cars/
  └── your_dataset/
```

Note that directory names are all **lowercase**.

If you want to change the data root, please set the argument `--data_root`.

### About `folder` format datasets

The `folder` format dataset is commonly used in image classification and can be loaded by `ImageFolder` implemented
in [`qtcls/datasets/folder.py`](../qtcls/datasets/folder.py). `ImageNet` is of the typical `folder` format. The dataset
folder/directory contains the split subfolders (`train`, `val`, etc.). For each split, images belonging to a particular
category are stored in a subfolder with the same name as the category. Here is an example:

```
dataset_name/
    ├── train/
        ├── category1/
            ├── image1
            ├── image2
            ├── image3
            ├── ...
        ├── category2/
            ├── image1
            ├── image2
            ├── image3
            ├── ...  
        ├── category3/ 
        ├── .../
    ├── val/
        ├── category1/
            ├── image1
            ├── image2
            ├── image3
            ├── ...
        ├── category2/
            ├── image1
            ├── image2
            ├── image3
            ├── ...  
        ├── category3/ 
        ├── .../
    ├── .../
```
