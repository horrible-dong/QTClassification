# 如何放置你的数据集

[English](README.md) | 简体中文

**默认的 `--data_root` 是 `./data`。 把你的数据集放在该目录下。**

每个数据集用单独的文件夹存储，例如：

```
data/
  ├── mnist/
  ├── cifar10/
  ├── cifar100/
  ├── pets/
  ├── imagenet/
  └── your_dataset/
```

注意：数据集文件夹名称*全部小写*。

如果你想改变存放数据集的根目录，请修改 `--data_root` 参数。