# How to write and import your configs

> Author: QIU, Tian  
> Affiliate: Zhejiang University  
> English | [简体中文](README_zh-CN.md)

If needed, refer to [`_demo_.py`](_demo_.py) and write your arguments in a config file (.py).

Since v0.2.0, when using your config file, set `--config` / `-c` to your **_config file path_**,
such as `configs/_demo_.py`.

For example,

```bash
python main.py --config configs/_demo_.py
```

or

```bash
python main.py -c configs/_demo_.py
```

Note that `--config` / `-c` supports any file system path, such as `configs/_demo_.py`,
`D:\\QTClassification\\configs\\_demo_.py`, `../../other_project/cfg.py`.

Then, the config arguments will be merged with the command line arguments `args` in [`main.py`](../main.py).
**Config arguments _override_ command line arguments if the name is duplicated.**