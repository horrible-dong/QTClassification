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
        # -- Method 1 --
        # Specify `optimizer_kwargs` in the config, e.g., `optimizer_kwargs=dict(lr=1e-4, weight_decay=5e-2, momentum=0.9)`
        return RMSprop(params, **args.optimizer_kwargs)

        # -- Method 2 --
        # return RMSprop(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    raise ValueError(f"Optimizer '{optimizer_name}' is not found.")
