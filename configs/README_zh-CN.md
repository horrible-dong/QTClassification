# 如何编写你的配置文件

[English](README.md) | 简体中文

如果需要的话，仿照 [`__demo__.py`](_demo_.py) 编写你的参数，最后这些参数会和 [`main.py`](../main.py) 中的命令行参数 `args`
进行合并。如果参数名相同，配置文件参数会覆盖命令行参数。

当使用你的配置文件时，把 `--config` 赋值为你的配置文件 ** **模块路径** ** ，如 `configs._demo_`（这个后续会优化）。

样例：

```bash
python main.py --config configs._demo_
```

或

```bash
python main.py -c configs._demo_
```