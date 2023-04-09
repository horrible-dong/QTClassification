# How to put your dataset

English | [简体中文](README_zh-CN.md)

**The default `--data_root` is `./data`. Put your dataset in this directory.**

Each dataset is stored in a separate directory. For example,

```
data/
  ├── mnist/
  ├── cifar10/
  ├── cifar100/
  ├── imagenet1k/
  ├── stl10/
  ├── svhn/
  ├── pets/
  └── your_dataset/
```

Note that directory names are all **lowercase**.

If you want to change the data root, please set the argument `--data_root`.
