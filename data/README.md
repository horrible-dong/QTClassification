# How to put your dataset

English | [简体中文](README_zh-CN.md)

**The default `--data_root` is `./data`. Put your dataset in this directory.**

Each dataset is stored in a separate directory. For example,

```
data/
  ├── mnist/
  ├── cifar10/
  ├── cifar100/
  ├── pets/
  ├── imagenet/
  └── your_dataset/
```

Note that directory names are all _lowercase_.

If you want to change the data root, please set the argument `--data_root`.
