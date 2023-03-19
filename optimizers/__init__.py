# Copyright (c) QIU, Tian. All rights reserved.

from torch.optim import *


def build_optimizer(args, params):
    optimizer_name = args.optimizer.lower()

    if optimizer_name in ['sgd', 'default']:
        return SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if optimizer_name == 'adam':
        return Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    raise ValueError(f"optimizer '{optimizer_name}' is not found.")
