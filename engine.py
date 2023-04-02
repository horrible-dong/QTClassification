# Copyright (c) QIU, Tian. All rights reserved.

import math
import sys
from typing import Iterable

import torch

from qtcls import build_evaluator
from qtcls.utils.misc import update, reduce_dict, MetricLogger, SmoothedValue


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, max_norm: float = 0,
                    scaler: torch.cuda.amp.GradScaler = None, print_freq: int = 10, need_targets: bool = False):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if need_targets:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        update(optimizer, losses, model, max_norm, scaler)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args, print_freq=10, need_targets=False):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = build_evaluator(args)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = targets.to(device)

        if need_targets:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        evaluator.update(outputs, targets)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    evaluator.synchronize_between_processes()
    evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    stats['eval'] = list(evaluator.eval.values())

    return stats, evaluator
