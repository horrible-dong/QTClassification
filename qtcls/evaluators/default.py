# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['DefaultEvaluator']

import itertools
import warnings
from typing import List

from sklearn import metrics as sklearn_metrics

from ..utils.misc import all_gather

warnings.filterwarnings('ignore')


class DefaultEvaluator:
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.outputs = []
        self.targets = []
        self.eval = {metric: None for metric in metrics}

    def update(self, outputs, targets):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'update(self, outputs, targets)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs['logits']
        outputs = outputs.max(1)[1].tolist()
        targets = targets.tolist()
        self.outputs += outputs
        self.targets += targets

    def synchronize_between_processes(self):
        self.outputs = list(itertools.chain(*all_gather(self.outputs)))
        self.targets = list(itertools.chain(*all_gather(self.targets)))

    @staticmethod
    def metric_acc(outputs, targets, **kwargs):
        return sklearn_metrics.accuracy_score(targets, outputs)

    @staticmethod
    def metric_recall(outputs, targets, **kwargs):
        return sklearn_metrics.recall_score(targets, outputs, average='macro')

    @staticmethod
    def metric_precision(outputs, targets, **kwargs):
        return sklearn_metrics.precision_score(targets, outputs, average='macro')

    @staticmethod
    def metric_f1(outputs, targets, **kwargs):
        return sklearn_metrics.f1_score(targets, outputs, average='macro')

    def summarize(self):
        print('Classification Metrics:')
        for metric in self.metrics:
            value = getattr(self, f'metric_{metric}')(self.outputs, self.targets)
            self.eval[metric] = value
            print(f'{metric}: {value:.3f}', end='    ')
        print('\n')
