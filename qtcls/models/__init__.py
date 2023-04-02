# Copyright (c) QIU, Tian. All rights reserved.

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
from .swin_transformer import *
from .vgg import *
from .vision_transformer_timm import *
from .vision_transformer_torchvision import *

__vars__ = vars()


def build_model(args):
    import torch
    from torch.hub import load_state_dict_from_url
    from .. import datasets
    from ..utils.io import checkpoint_loader
    from ..utils.misc import is_main_process

    model_lib = args.model_lib.lower()
    model_name = args.model.lower()
    num_classes = datasets.num_classes[args.dataset.lower()]
    pretrained = not args.no_pretrain and is_main_process()

    if model_lib == 'torchvision-ex':
        if __vars__.get(model_name):
            model = __vars__[model_name](num_classes=num_classes)
        else:
            raise KeyError(f"model '{model_name}' is not found.")

        if pretrained:
            found_local_path = search_pretrained_from_local_paths(model_name)
            found_url = search_pretrained_from_urls(model_name)

            if found_local_path:
                state_dict = torch.load(found_local_path)
            elif found_url:
                state_dict = load_state_dict_from_url(found_url, progress=True)
            else:
                raise FileNotFoundError(f"pretrained model for '{model_name}' is not found")

            checkpoint_loader(model, state_dict, strict=False)

        return model

    if model_lib == 'timm':
        import timm
        return timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"model_lib '{model_lib}' is not found.")


def search_pretrained_from_local_paths(model_name):
    import os
    from ._pretrain_ import model_local_paths
    found_local_path = None
    if model_local_paths.get(model_name):
        local_paths = model_local_paths[model_name]
        if isinstance(local_paths, str):
            local_paths = [local_paths]
        for path in local_paths:
            if os.path.exists(path):
                found_local_path = path
                break
    return found_local_path


def search_pretrained_from_urls(model_name):
    from ._pretrain_ import model_urls
    found_url = None
    if model_urls.get(model_name):
        found_url = model_urls[model_name]
    return found_url
