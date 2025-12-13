# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['MiniDataLoader']

import torch


class MiniDataLoader:
    def __init__(self, samples: list, targets: list, batch_size: int, shuffle: bool = False, drop_last: bool = False):
        self.samples = samples
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        samples, targets = self.samples, self.targets

        if self.shuffle:
            shuffled_indices = torch.randperm(len(self.samples))
            samples, targets = samples[shuffled_indices], targets[shuffled_indices]

        start_index, end_index = 0, 0

        for i in range(len(samples) // self.batch_size):
            start_index, end_index = i * self.batch_size, (i + 1) * self.batch_size
            yield samples[start_index: end_index], targets[start_index: end_index]

        if len(samples) % self.batch_size != 0 and not self.drop_last:
            yield samples[end_index:], targets[end_index:]

    def __len__(self):
        length = len(self.samples)
        if self.drop_last:
            return length // self.batch_size
        else:
            from math import ceil
            return ceil(length / self.batch_size)
