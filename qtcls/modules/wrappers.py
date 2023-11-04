# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['SingleInputModuleWrapper', 'SingleInputModuleListWrapper', 'MultiInputModuleListWrapper']

from torch import nn


class SingleInputModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x)


class SingleInputModuleListWrapper(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x)
        return x


class MultiInputModuleListWrapper(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x
