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
              n_cols=4,
              ewm_com=2,
              drop_cols=('epoch', 'n_params'),
              log_name='log.txt',
              output_file='./plot_logs.pdf'):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list of paths, each pointing to individual dir with a log file, or a single path.
              - n_cols = number of columns in the plots.
              - ewm_com = specify the degree of the exponential weighted smoothing of the plots.
              - drop_cols = columns to drop from the plots.
              - log_name = name of the log file.
              - output_file = path of the output file.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = 'log_plot.py::plot_logs'

    if not isinstance(logs, list):
        if isinstance(logs, str):
            logs = [logs]
        else:
            raise TypeError(f'{func_name} - invalid argument for logs parameter.\n \
            Expect list[str] or single str, received {type(logs)}')

    if len(logs) == 0:
        return

    for dir in logs:
        if not os.path.exists(dir):
            print(f'--> missing directory: {dir}')
            continue
        fn = os.path.join(dir, log_name)
        if not os.path.exists(fn):
            print(f'--> missing log file: {fn}')

    dfs = [pd.read_json(os.path.join(p, log_name), lines=True).drop(list(drop_cols), axis=1, errors='ignore')
           for p in logs]

    fields = _merge_train_test_fields(dfs[0].columns)

    n_rows = math.ceil(len(fields) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        df = df.ewm(com=ewm_com).mean()
        for n, field in enumerate(fields):
            i, j = n // n_cols, n % n_cols
            df.plot(y=field,
                    ax=axs[i, j],
                    color=[color] * len(field),
                    style=['-', '--', '-.', ':'][:len(field)])

    for ax, field in zip(axs.ravel(), fields):
        if len(field) == 1:
            ax.legend([os.path.basename(p) for p in logs], loc=0)
        elif len(field) == 2:
            ax.legend([f'{os.path.basename(p)}_{suffix}' for p in logs for suffix in ['train', 'test']], loc=0)
        else:
            ...
        ax.set_title(field[0].replace('train_', '').replace('test_', ''))

    if os.path.exists(output_file):
        os.remove(output_file)
    plt.savefig(output_file)
    plt.close(fig)


def _merge_train_test_fields(fields):
    res, merged = [], []
    for field in fields:
        if field in merged:
            continue
        elif field.startswith('train_') and field.replace('train_', 'test_', 1) in fields:
            res.append([field, field.replace('train_', 'test_', 1)])
            merged.append(field.replace('train_', 'test_', 1))
        elif field.startswith('test_') and field.replace('test_', 'train_', 1) in fields:
            res.append([field.replace('test_', 'train_', 1), field])
            merged.append(field.replace('test_', 'train_', 1))
        else:
            res.append([field])

    return res


if __name__ == '__main__':
    # Each path points to individual dir with a log file
    plot_logs([

        # r'/path/to/your/dir_1',
        # r'/path/to/your/dir_2',
        # r'/path/to/your/dir_3',

    ], output_file='./plot_logs.pdf')
