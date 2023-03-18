# Copyright (c) QIU, Tian. All rights reserved.

import itertools

from sklearn import metrics as sklearn_metrics

from utils.misc import all_gather


class DefaultEvaluator:
    def __init__(self, metrics: list):
        self.metrics = metrics
        self.outputs = []
        self.targets = []

    def update(self, outputs, targets):
        outputs = outputs.max(1)[1].tolist()
        targets = targets.tolist()
        self.outputs += outputs
        self.targets += targets

    def synchronize_between_processes(self):
        self.outputs = list(itertools.chain(*all_gather(self.outputs)))
        self.targets = list(itertools.chain(*all_gather(self.targets)))

    def metric_acc(self, outputs, targets, **kwargs):
        return sklearn_metrics.accuracy_score(targets, outputs) * 100

    def metric_recall(self, outputs, targets, **kwargs):
        return sklearn_metrics.recall_score(targets, outputs, average='macro') * 100

    def metric_precision(self, outputs, targets, **kwargs):
        return sklearn_metrics.precision_score(targets, outputs, average='macro') * 100

    def metric_f1(self, outputs, targets, **kwargs):
        return sklearn_metrics.f1_score(targets, outputs, average='macro') * 100

    def summarize(self):
        print('Classification Metrics:')
        for metric in self.metrics:
            print("{}: {:.2f}".format(metric, getattr(self, f'metric_{metric}')(self.outputs, self.targets)), end='   ')
        print('\n')
