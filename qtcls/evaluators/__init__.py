# Copyright (c) QIU Tian. All rights reserved.

from .default import DefaultEvaluator


def build_evaluator(args):
    evaluator_name = args.evaluator.lower()

    if evaluator_name in ['default']:
        return DefaultEvaluator(metrics=['acc', 'recall', 'precision', 'f1'])

    raise ValueError(f"Evaluator '{evaluator_name}' is not found.")
