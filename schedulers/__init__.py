# Copyright (c) QIU, Tian. All rights reserved.

from .cosine import CosineLR
from .warmup_multistep import WarmupMultiStepLR


def build_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()

    if scheduler_name in ['cosine', 'default']:
        return CosineLR(optimizer, args.epochs, args.lrf)

    raise ValueError(f"scheduler '{scheduler_name}' is not found.")
