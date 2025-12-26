# How to write and import your configs

> Author: QIU Tian  
> Affiliation: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

## Basic Tutorial

If needed, refer to [`_demo_.py`](_demo_.py) and write your arguments in a config file (.py).

Starting from v0.2.0, if you want to use your config file, set `--config` / `-c` to your **_config file path_**,
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
`D:\\QTClassification\\configs\\_demo_.py`, `../../another_project/cfg.py`.

Then, the config arguments **override or merge** with the command-line arguments `args` pre-defined in
[`main.py`](../main.py).

### Important Feature

Starting from v0.7.0, command-line arguments **_after_** `--config xxx` / `-c xxx` **override** the config arguments if
the name is duplicated. For example,

```bash
python main.py -c configs/_demo_.py -co  # clear the output dir first
python main.py -c configs/_demo_.py --batch_size 100 --print_freq 200 --note bs100
python main.py -c configs/_demo_.py --save_interval 5555  # do not save
python main.py -c configs/_demo_.py --dataset food --dummy  # use fake data
python main.py -c configs/_demo_.py -d cifar100 -b 400 --note cifar100-bs400
python main.py -c configs/_demo_.py --resume ./runs/cifar10/vit_tiny_patch4_32/checkpoint.pth
python main.py -c configs/_demo_.py --resume ./runs/cifar10/vit_tiny_patch4_32/checkpoint.pth --eval
```

## Advanced Tutorial

We will provide it in the v1.0.0 release. Stay tuned.
