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
from termcolor import cprint

from qtcls.utils.decorators import main_process_only


@main_process_only
def plot_logs(log_dirs,
              n_cols=4,
              ewm_com=0,
              drop_cols=('epoch', 'n_params'),
              log_name='log.txt',
              output_file='./plot_logs.pdf'):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - log_dirs = list of paths, each pointing to an individual dir with a log file, or a single path.
              - n_cols = number of columns in the plots.
              - ewm_com = specify the degree of the exponential weighted smoothing of the plots.
              - drop_cols = columns to drop from the plots.
              - log_name = name of the log file.
              - output_file = path of the output file.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = 'log_plot.py::plot_logs'

    if not isinstance(log_dirs, list):
        if isinstance(log_dirs, str):
            log_dirs = [log_dirs]
        else:
            raise TypeError(f"{func_name} - expected 'log_dirs' to be list[str] or str, "
                            f"but got {type(log_dirs).__name__}")

    _log_dirs = []
    for log_dir in log_dirs:
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            cprint(f'{func_name} - missing log dir: {log_dir}', 'light_yellow')
        elif not os.path.exists(log_file):
            cprint(f'{func_name} - missing log file: {log_file}', 'light_yellow')
        else:
            _log_dirs.append(log_dir)
    log_dirs = _log_dirs

    if len(log_dirs) == 0:
        cprint(f'{func_name} - no log files found', 'light_yellow')
        return

    dfs = [pd.read_json(os.path.join(log_dir, log_name), lines=True).drop(list(drop_cols), axis=1, errors='ignore')
           for log_dir in log_dirs]

    for df, log_dir in zip(dfs, log_dirs):
        if df.columns.tolist() != dfs[-1].columns.tolist():
            cprint(f'{func_name} - inconsistent fields between {log_dir} and {log_dirs[-1]} (ref.)', 'light_yellow')

    fields = _merge_train_test_fields(dfs[-1].columns)

    n_rows = math.ceil(len(fields) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    field_to_log_dirs = {tuple(field): [] for field in fields}

    for log_dir, df, color in zip(log_dirs, dfs, sns.color_palette(n_colors=len(log_dirs))):
        df = df.ewm(com=ewm_com).mean()
        for n, field in enumerate(fields):
            if all(f in df.columns for f in field):
                i, j = n // n_cols, n % n_cols
                df.plot(y=field,
                        ax=axs[i, j],
                        color=[color] * len(field),
                        style=['-', '--', '-.', ':'][:len(field)])
                field_to_log_dirs[tuple(field)].append(log_dir)

    for ax, field in zip(axs.ravel(), fields):
        if len(field) == 1:
            ax.legend([os.path.basename(log_dir)
                       for log_dir in field_to_log_dirs[tuple(field)]], loc=0)
        elif len(field) == 2:
            ax.legend([f'{os.path.basename(log_dir)}_{t}'
                       for log_dir in field_to_log_dirs[tuple(field)] for t in ['train', 'test']], loc=0)
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
    # Each path points to an individual dir with a log file.
    plot_logs([

        # r'/path/to/dir_1',
        # r'/path/to/dir_2',
        # r'/path/to/dir_3',

    ], output_file='./plot_logs.pdf')
