# -------------------------------------------------------------------------
# Modified by QIU Tian
# Copyright (c) QIU Tian. All rights reserved.
# -------------------------------------------------------------------------
# https://github.com/facebookresearch/detr/blob/main/util/plot_utils.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------------

"""
Plotting utilities to visualize training logs.
"""

import math
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt


def plot_logs(logs,
              fields=('loss', 'loss_ce_unscaled',
                      'Accuracy', 'Recall', 'Precision', 'F1-Score'),
              ewm_col=0,
              log_name='log.txt',
              n_cols=3,
              output_file='./plot_logs.pdf'):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "log_plot.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    n_rows = math.ceil(len(fields) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows))

    cls_fields = fields[-4:]
    cls_metrics = {field: i for i, field in enumerate(cls_fields)}

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for n, field in enumerate(fields):
            i, j = n // n_cols, n % n_cols
            if field in cls_fields:
                coco_eval = pd.DataFrame(
                    np.stack(df.test_eval.dropna().values)[:, cls_metrics[field]]
                ).ewm(com=ewm_col).mean()
                axs[i, j].plot(coco_eval, c=color, )
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[i, j],
                    color=[color] * 2,
                    style=['-', '--'],
                )

    for ax, field in zip(axs.ravel(), fields):
        if field in cls_fields:
            ax.legend([Path(p).name for p in logs], loc=4)
        else:
            legend = []
            for p in logs:
                legend.extend([f'{Path(p).name}_train', f'{Path(p).name}_test'])
            ax.legend(legend)
        ax.set_title(field)

    plt.savefig(output_file)
    # plt.close()
    # plt.show()


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


if __name__ == '__main__':
    # Each 'Path' points to individual dir with a log file
    plot_logs([

        # Path(r"/path/to/your/dir_1"),
        # Path(r"/path/to/your/dir_2"),
        # Path(r"/path/to/your/dir_3"),

    ], output_file='./plot_logs.pdf')
