# Copyright (c) QIU, Tian. All rights reserved.

from .cross_entropy import CrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def build_criterion(args):
    criterion_name = args.criterion.lower()

    if criterion_name == 'ce':
        losses = ['labels']
        weight_dict = {'loss_ce': 1}
        return CrossEntropy(losses=losses, weight_dict=weight_dict)

    if criterion_name == 'label_smoothing_ce':
        losses = ['labels']
        weight_dict = {'loss_ce': 1}
        return LabelSmoothingCrossEntropy(losses=losses, weight_dict=weight_dict, smoothing=args.label_smoothing)

    if criterion_name in ['soft_target_ce', 'default']:
        losses = ['labels']
        weight_dict = {'loss_ce': 1}
        return SoftTargetCrossEntropy(losses=losses, weight_dict=weight_dict)

    raise ValueError(f"Criterion '{criterion_name}' is not found.")
