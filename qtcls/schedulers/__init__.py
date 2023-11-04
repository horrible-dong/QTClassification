# Copyright (c) QIU Tian. All rights reserved.

import timm.scheduler as timm_scheduler
import torch.optim.lr_scheduler as tv_scheduler


def build_scheduler(args, optimizer, n_iter_per_epoch):
    scheduler_name = args.scheduler.lower()

    if scheduler_name == 'cosine':
        if args.warmup_epochs > 0 and args.warmup_steps > 0:
            raise AssertionError("'args.warmup_epochs' and 'args.warmup_steps' cannot both be positive.")
        num_steps = int(args.epochs * n_iter_per_epoch)
        warmup_steps = int(args.warmup_epochs * n_iter_per_epoch) if args.warmup_epochs > 0 else args.warmup_steps
        return timm_scheduler.CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps),
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    if scheduler_name == 'step':
        return tv_scheduler.StepLR(optimizer, args.step_size, args.gamma)

    if scheduler_name == 'multistep':
        return timm_scheduler.MultiStepLRScheduler(
            optimizer,
            decay_t=args.milestones,
            decay_rate=args.gamma,
            warmup_t=args.warmup_epochs,
            warmup_lr_init=args.warmup_lr,
            t_in_epochs=True,
        )
        # return tv_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    raise ValueError(f"Scheduler '{scheduler_name}' is not found.")
