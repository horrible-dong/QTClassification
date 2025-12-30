# 如何编写和导入你的配置文件

> 作者: QIU Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

## 基本教程

参考 [`_demo_.py`](_demo_.py)，将你的参数写入配置文件（.py）中。

从 v0.2.0 开始，使用配置文件时，把 `--config` / `-c` 赋值为你的配置文件路径，如 `configs/_demo_.py`：

```bash
# 全写
python main.py --config /path/to/config.py

# 简写
python main.py -c /path/to/config.py

# 样例
python main.py -c configs/_demo_.py
```

值得注意的是，`--config` / `-c` 支持任意文件系统路径，比如
`configs/_demo_.py`，`/root/QTClassification/configs/_demo_.py`，`../../another_project/cfgs/exp_name.py`。

然后，配置文件参数会**覆盖或合并** [`main.py`](../main.py) 中预定义的命令行参数 `args`。

### 重要特性

从 v0.7.0 开始，`--config xxx` / `-c xxx` 之后的命令行参数会覆盖配置文件参数（参数名相同时），例如：

```bash
python main.py -c configs/_demo_.py -co  # 预先清空输出目录
python main.py -c configs/_demo_.py --batch_size 100 --print_freq 200 --note bs100
python main.py -c configs/_demo_.py --save_interval 5555  # 不保存
python main.py -c configs/_demo_.py --dataset food --dummy  # 使用假数据
python main.py -c configs/_demo_.py -d cifar100 -b 400 --note cifar100-bs400
python main.py -c configs/_demo_.py --resume ./runs/cifar10/vit_tiny_patch4_32/checkpoint.pth
python main.py -c configs/_demo_.py --resume ./runs/cifar10/vit_tiny_patch4_32/checkpoint.pth --eval
```

同理，`--config xxx` / `-c xxx` 之前的命令行参数会被配置文件参数覆盖（参数名相同时）。

## 进阶教程

我们会在 v1.0.0 版本中给出，敬请期待。
