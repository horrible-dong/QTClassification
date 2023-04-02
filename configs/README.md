# How to edit your configs

English | [简体中文](README_zh-CN.md)

If needed, refer to [`__demo__.py`](_demo_.py) and write your arguments. Finally, these arguments will be merged with
the
command line arguments `args` in [`main.py`](../main.py). **Configuration arguments _override_ command line arguments if
the
names are the same.**

When using your configuration file, set `--config` to your configuration file **_module path_**, such
as `configs._demo_` (this will be optimized later).

For example,

```bash
python main.py --config configs._demo_
```

or

```bash
python main.py -c configs._demo_
```