# Copyright (c) QIU, Tian. All rights reserved.

import itertools

from sklearn import metrics as sklearn_metrics

from ..utils.misc import all_gather


class DefaultEvaluator:
    def __init__(self, metrics: list):
        self.metrics = metrics
        self.outputs = []
        self.targets = []
        self.eval = {metric: None for metric in metrics}

    def update(self, outputs, targets):
        if isinstance(outputs, dict):
            if outputs.get('logits') is not None:
                outputs = outputs["logits"]
            else:
                outputs = list(outputs.values())[0]
        outputs = outputs.max(1)[1].tolist()
        targets = targets.tolist()
        self.outputs += outputs
        self.targets += targets

    def synchronize_between_processes(self):
        self.outputs = list(itertools.chain(*all_gather(self.outputs)))
        self.targets = list(itertools.chain(*all_gather(self.targets)))

    def metric_acc(self, outputs, targets, **kwargs):
        return sklearn_metrics.accuracy_score(targets, outputs)

    def metric_recall(self, outputs, targets, **kwargs):
        return sklearn_metrics.recall_score(targets, outputs, average='macro')

    def metric_precision(self, outputs, targets, **kwargs):
        return sklearn_metrics.precision_score(targets, outputs, average='macro')  # TODO remove warning

    def metric_f1(self, outputs, targets, **kwargs):
        return sklearn_metrics.f1_score(targets, outputs, average='macro')

    def summarize(self):
        print('Classification Metrics:')
        for metric in self.metrics:
            value = getattr(self, f'metric_{metric}')(self.outputs, self.targets)
            self.eval[metric] = value
            print("{}: {:.3f}".format(metric, value), end='    ')
        print('\n')
