# Copyright (c) QIU, Tian. All rights reserved.

import torch
from torch.hub import load_state_dict_from_url

import datasets
from utils.io import checkpoint_loader
from utils.misc import is_main_process
from ._pretrain_ import model_local_paths, model_urls
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
        if __vars__.get(model_name):
            model = __vars__[model_name](num_classes=num_classes)
        else:
            raise KeyError(f"model '{model_name}' is not found.")

        if pretrained:
            if model_local_paths.get(model_name):
                state_dict = torch.load(model_local_paths[model_name])
            elif model_urls.get(model_name):
                state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
            else:
                raise FileNotFoundError(f"pretrained model for '{model_name}' is not found")
            checkpoint_loader(model, state_dict, strict=False)

        return model

    if model_lib == 'timm':
        import timm
        return timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"model_lib '{model_lib}' is not found.")
