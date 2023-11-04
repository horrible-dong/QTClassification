# Copyright (c) QIU Tian. All rights reserved.

from torch.optim import *


def build_optimizer(args, params):
    optimizer_name = args.optimizer.lower()

    if optimizer_name == 'sgd':
        return SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if optimizer_name == 'adam':
        return Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    if optimizer_name == 'adamw':
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

    if optimizer_name == 'rmsprop':
        return RMSprop(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    raise ValueError(f"Optimizer '{optimizer_name}' is not found.")
