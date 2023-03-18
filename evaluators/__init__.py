# Copyright (c) QIU, Tian. All rights reserved.

from .default import DefaultEvaluator
from .metric_logger import MetricLogger


def build_evaluator(args):
    evaluator_name = args.evaluator.lower()

    if evaluator_name in ['default']:
        metrics = ['acc', 'recall', 'precision', 'f1']
        return DefaultEvaluator(metrics)

    raise ValueError(f'{evaluator_name} is not exist.')
