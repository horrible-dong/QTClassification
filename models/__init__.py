# Copyright (c) QIU, Tian. All rights reserved.

import datasets
from utils.decorators import getattr_case_insensitive
from utils.misc import is_main_process
from .alexnet import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .googlenet import *
from .inception import *
from .mnasnet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .regnet import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .vgg import *
from .vision_transformer import *

__vars__ = vars()


def build_model(args):
    model_name = args.model.lower()
    return __vars__[model_name](num_classes=get_num_classes(args.dataset),
                                pretrained=is_main_process() and not args.no_pretrain)


@getattr_case_insensitive
def get_num_classes(dataset_name):
    return len(getattr(datasets, dataset_name).classes)
