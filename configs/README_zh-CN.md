# 如何编写和导入你的配置文件

> 作者: QIU, Tian  
> 机构: 浙江大学  
> [English](README.md) | 简体中文

如果需要的话，仿照 [`_demo_.py`](_demo_.py) 编写你的参数。

从 v0.2.0 开始，当使用你的配置文件时，把 `--config` / `-c` 赋值为你的 **_配置文件路径_**，如 `configs/_demo_.py`。

样例：

```bash
python main.py --config configs/_demo_.py
```

或

```bash
python main.py -c configs/_demo_.py
```

值得注意的是，`--config` / `-c` 可以支持任意文件系统路径，比如
`configs/_demo_.py`, `D:\\QTClassification\\configs\\_demo_.py`, `../../other_project/cfg.py`。

然后，配置文件参数会和 [`main.py`](../main.py) 中的命令行参数 `args` 进行**覆盖或合并**。

从 v0.7.0 开始，`--config xxx` / `-c xxx` **_之后_** 的命令行参数会**覆盖**配置文件参数（参数名相同时）。
