# How to write and import your configs

> Author: QIU Tian  
> Affiliation: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

## Basic Tutorial

Refer to [`_demo_.py`](_demo_.py) and write your arguments in a config file (.py).

Starting from v0.2.0, when using your config file, set `--config` / `-c` to your config file path,
like `configs/_demo_.py`:

```bash
# full command
python main.py --config /path/to/config.py

# short command
python main.py -c /path/to/config.py

# example
python main.py -c configs/_demo_.py
```

Note that `--config` / `-c` supports any file system path, such as `configs/_demo_.py`,
`/root/QTClassification/configs/_demo_.py`, `../../another_project/cfgs/exp_name.py`.

Then, the config arguments **override or merge** with the command-line arguments `args` pre-defined in
[`main.py`](../main.py).

### Important Features

Starting from v0.7.0, command-line arguments after `--config xxx` / `-c xxx` override the config arguments
if the name is duplicated. For example,

```bash
python main.py -c configs/_demo_.py -co  # clear the output dir first
python main.py -c configs/_demo_.py --batch_size 100 --print_freq 200 --note bs100
python main.py -c configs/_demo_.py --save_interval 5555  # do not save
python main.py -c configs/_demo_.py --dataset food --dummy  # use fake data
python main.py -c configs/_demo_.py -d cifar100 -b 400 --note cifar100-bs400
python main.py -c configs/_demo_.py --resume ./runs/cifar10/vit_tiny_patch4_32/checkpoint.pth
python main.py -c configs/_demo_.py --resume ./runs/cifar10/vit_tiny_patch4_32/checkpoint.pth --eval
```

Similarly, command-line arguments before `--config xxx` / `-c xxx` are overridden by the config arguments
if the name is duplicated.

## Advanced Tutorial

We will provide it in the v1.0.0 release. Stay tuned.
