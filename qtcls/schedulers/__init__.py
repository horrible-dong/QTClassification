# Copyright (c) QIU, Tian. All rights reserved.

from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import *


def build_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()

    if scheduler_name == 'cosine':
        return CosineLRScheduler(
            optimizer,
            t_initial=args.epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
        )

    if scheduler_name == 'step':
        return StepLR(optimizer, args.step_size, args.gamma)

    if scheduler_name == 'multistep':
        return MultiStepLR(optimizer, args.milestones, args.gamma)

    raise ValueError(f"scheduler '{scheduler_name}' is not found.")
