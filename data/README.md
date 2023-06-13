# How to put your dataset

> Author: QIU, Tian  
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
