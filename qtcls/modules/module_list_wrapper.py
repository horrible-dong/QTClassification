# Copyright (c) QIU, Tian. All rights reserved.

from torch import nn


class ModuleListWrapper(nn.ModuleList):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x
