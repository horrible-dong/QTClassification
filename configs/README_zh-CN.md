# 如何编写和导入你的配置文件

[English](README.md) | 简体中文

如果需要的话，仿照 [`_demo_.py`](_demo_.py) 编写你的参数。

从 v0.2.0 开始，当使用你的配置文件时，把 `--config` 赋值为你的 **_配置文件路径_**，如 `configs/_demo_.py`。 

样例：

```bash
python main.py --config configs/_demo_.py
```

或

```bash
python main.py -c configs/_demo_.py
```

值得注意的是，`--config` 可以支持任意文件系统路径，比如 `configs/_demo_.py`, `D:\\QTClassification\\configs\\_demo_.py`,
`../../other_project/cfg.py`。

然后，这些参数会和 [`main.py`](../main.py) 中的命令行参数 `args` 进行合并。**如果参数名相同，配置文件参数会覆盖命令行参数。**