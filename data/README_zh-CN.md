# 如何放置你的数据集

> 作者: QIU Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

**默认的 `--data_root` 为 `./data`。 把你的数据集放在该目录下。**

每个数据集用单独的文件夹存储，例如：

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

注意：数据集文件夹名称**全部小写**。

如果你想改变存放数据集的根目录，请修改 `--data_root` 参数。

### 关于 `folder` 格式数据集

`folder` 格式数据集常用于图像分类，可以使用 [`qtcls/datasets/folder.py`](../qtcls/datasets/folder.py)
中的 `ImageFolder` 类进行读取。`ImageNet` 是典型的 `folder` 格式数据集。数据集文件夹/目录包含 split
子文件夹（`train`、`val` 等）。对于每个 split，同一类别的图像存储在与该类别同名的子文件夹中。这是一个样例：

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

