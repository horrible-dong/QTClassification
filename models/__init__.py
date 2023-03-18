# Copyright (c) QIU, Tian. All rights reserved.

import datasets
from utils.decorators import getattr_case_insensitive
from .vit import *


@getattr_case_insensitive
def build_model(args):
    model_name = args.model.lower()

    if model_name == 'vit-b16-224':
        return vit_base_patch16_224(num_classes=len(getattr(datasets, args.dataset).classes))

    raise ValueError(f'{model_name} is not exist.')
