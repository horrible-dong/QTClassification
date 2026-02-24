# -------------------------------------------------------------------------
# Modified by QIU Tian
# Copyright (c) QIU Tian. All rights reserved.
# -------------------------------------------------------------------------
# https://github.com/facebookresearch/detr/blob/main/util/plot_utils.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -------------------------------------------------------------------------

"""
Plotting utilities to visualize training logs.
"""

import math
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from qtcls.utils.decorators import main_process_only


@main_process_only
def plot_logs(logs,
              loss_fields=('loss', 'loss_ce_unscaled'),
              metric_fields=('Accuracy', 'Recall', 'Precision', 'F1-Score'),
              ewm_com=0,
              n_cols=3,
              log_name='log.txt',
              output_file='./plot_logs.pdf'):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list of paths, each pointing to individual dir with a log file, or a single path.
              - loss_fields = which results to plot from both training and test losses of each log file.
              - metric_fields = which results to plot from the `test_eval` list of each log file.
              - ewm_com = specify the degree of the exponential weighted smoothing of the plots.
              - n_cols = number of columns in the plots.
              - log_name = name of the log file.
              - output_file = path of the output file.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "log_plot.py::plot_logs"

    if not isinstance(logs, list):
        if isinstance(logs, str):
            logs = [logs]
        else:
            raise TypeError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[str] or single str, received {type(logs)}")

    for dir in logs:
        if not os.path.exists(dir):
            print(f"--> missing directory: {dir}")
            continue
        fn = os.path.join(dir, log_name)
        if not os.path.exists(fn):
            print(f"--> missing log file: {fn}")

    # load log file(s) and plot
    dfs = [pd.read_json(os.path.join(p, log_name), lines=True) for p in logs]

    fields = loss_fields + metric_fields

    n_rows = math.ceil(len(fields) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows))

    metrics = {field: i for i, field in enumerate(metric_fields)}

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for n, field in enumerate(fields):
            i, j = n // n_cols, n % n_cols
            if field in metric_fields:
                df['test_eval'].apply(pd.Series)[metrics[field]].ewm(com=ewm_com).mean().plot(
                    ax=axs[i, j],
                    color=color
                )
            else:
                df.drop('test_eval', axis=1).ewm(com=ewm_com).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[i, j],
                    color=[color, color],
                    style=['-', '--']
                )

    for ax, field in zip(axs.ravel(), fields):
        if field in metric_fields:
            ax.legend([os.path.basename(p) for p in logs], loc=4)
        else:
            legend = []
            for p in logs:
                legend.extend([f'{os.path.basename(p)}_train', f'{os.path.basename(p)}_test'])
            ax.legend(legend)
        ax.set_title(field)

    plt.savefig(output_file)
    # plt.close()
    # plt.show()


if __name__ == '__main__':
    # Each path points to individual dir with a log file
    plot_logs([

        # r"/path/to/your/dir_1",
        # r"/path/to/your/dir_2",
        # r"/path/to/your/dir_3",

    ], output_file='./plot_logs.pdf')
