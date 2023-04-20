# Copyright (c) QIU, Tian. All rights reserved.

from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import *


def build_scheduler(args, optimizer, n_iter_per_epoch):
    scheduler_name = args.scheduler.lower()

    if scheduler_name == 'cosine':
        if args.warmup_epochs > 0 and args.warmup_steps > 0:
            raise AssertionError("'args.warmup_epochs' and 'args.warmup_steps' cannot be positive at the same time.")
        num_steps = int(args.epochs * n_iter_per_epoch)
        warmup_steps = int(args.warmup_epochs * n_iter_per_epoch) if args.warmup_epochs > 0 else args.warmup_steps
        return CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps),
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    if scheduler_name == 'step':
        return StepLR(optimizer, args.step_size, args.gamma)

    if scheduler_name == 'multistep':
        return MultiStepLR(optimizer, args.milestones, args.gamma)

    raise ValueError(f"scheduler '{scheduler_name}' is not found.")
