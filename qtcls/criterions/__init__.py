# Copyright (c) QIU, Tian. All rights reserved.

from .default import DefaultCriterion


def build_criterion(args):
    criterion_name = args.criterion.lower()

    if criterion_name in ['ce', 'default']:
        losses = ['labels']
        weight_dict = {'loss_ce': 1}
        return DefaultCriterion(losses=losses, weight_dict=weight_dict)

    raise ValueError(f"criterion '{criterion_name}' is not found.")
