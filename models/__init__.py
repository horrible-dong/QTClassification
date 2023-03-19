# Copyright (c) QIU, Tian. All rights reserved.

import datasets
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
    model_lib = args.model_lib.lower()
    model_name = args.model.lower()
    num_classes = datasets.num_classes[args.dataset.lower()]
    pretrained = not args.no_pretrain and is_main_process()

    if model_lib == 'torchvision':
        return __vars__[model_name](num_classes=num_classes, pretrained=pretrained)

    if model_lib == 'timm':
        import timm
        return timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f'model_lib {model_lib} is not exist.')
