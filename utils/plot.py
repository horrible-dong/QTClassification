# Copyright (c) QIU, Tian. All rights reserved.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils.io import json_loader


def plot_example(ys, xlabel, ylabel):
    # data
    # y = [i * 100 for i in y]
    legends = ["batch_size=16", "batch_size=32", "batch_size=64",
               "batch_size=128", "batch_size=256", "batch_size=512"]
    colors = ["red", "blue", "cyan", "purple", "orange", "green"]
    x = np.arange(1, 301)

    # label在legend中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color: b:blue, g:green, r:red, c:cyan, m:magenta, y:yellow, k:black, w:white ...
    # 线型: -  --   -.  :    ,
    # marker: .  ,   o   v    <    *    +    1
    linewidth = 4
    markersize = 14
    title_fontsize = 48
    xytick_fontsize = 20
    xylabel_fontsize = 20
    legend_fontsize = 20

    # 图像
    plt.figure(figsize=(16, 12))
    plt.gcf().subplots_adjust(bottom=0.20, left=0.16, right=0.965, top=0.84)

    # # 字体
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    # 标题
    # plt.title(title, fontsize=title_fontsize, fontweight='bold')  # 默认字体大小为12

    # 画图
    for y, legend, color in zip(ys, legends, colors):
        if legend != "batch_size=16":
            continue
        y = [i * 100 for i in y]
        plt.plot(x, y, color=color, label=legend, linewidth=linewidth, markersize=markersize)

    # for y, legend, color in zip([ys[0]], [legends[0]], [colors[0]]):
    #     # if legend != "batch_size=16":
    #     #     continue
    #     # y = [i * 100 for i in y]
    #     plt.plot(x, y, color=color, label=None, linewidth=linewidth, markersize=markersize)

    # 获取坐标轴
    ax = plt.gca()

    # 网格线
    # plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    # 坐标刻度
    plt.xticks(ticks=None, labels=None, fontsize=xytick_fontsize)
    plt.yticks(ticks=None, labels=None, fontsize=xytick_fontsize)

    # 坐标刻度字体大小
    plt.tick_params(labelsize=xytick_fontsize)

    # 坐标间隔
    x_major_locator = MultipleLocator(50)
    y_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    # 坐标标签
    plt.xlabel(xlabel, fontsize=xylabel_fontsize)
    plt.ylabel(ylabel, fontsize=xylabel_fontsize)

    # 坐标范围
    plt.xlim(-0.07, len(x) + 0.09)
    plt.ylim(-0.07, 100.1)

    # 图例
    plt.legend(loc=4, numpoints=1, ncol=1, frameon=False)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=legend_fontsize, fontweight='normal')  # 设置图例字体的大小和粗细

    # plt.savefig('./res.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def visualize(json_path, epochs):
    losses = []
    for b in ["16", "32", "64", "128", "256", "512"]:
        loss = json_loader(rf"D:\研究生\projects\Gridet\visualize\res-{b}.json")["accuracy"]
        for _ in range(300 - len(loss)):
            loss.append(loss[-1])
        losses.append(loss)

    plot_example(losses, xlabel="epochs", ylabel="Accuracy (%)")


if __name__ == '__main__':
    visualize(r"", 100)
