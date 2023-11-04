# Copyright (c) QIU Tian. All rights reserved.

from .cross_entropy import CrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def build_criterion(args):
    criterion_name = args.criterion.lower()

    if criterion_name == 'ce':
        return CrossEntropy(losses=['labels'], weight_dict={'loss_ce': 1})

    if criterion_name == 'label_smoothing_ce':
        return LabelSmoothingCrossEntropy(losses=['labels'], weight_dict={'loss_ce': 1}, smoothing=args.label_smoothing)

    if criterion_name in ['soft_target_ce', 'default']:
        return SoftTargetCrossEntropy(losses=['labels'], weight_dict={'loss_ce': 1})

    raise ValueError(f"Criterion '{criterion_name}' is not found.")
