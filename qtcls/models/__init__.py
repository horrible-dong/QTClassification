# Copyright (c) QIU, Tian. All rights reserved.

from .alexnet import *
from .cait import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .googlenet import *
from .inception import *
from .levit import *
from .mlp_mixer import *
from .mnasnet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .poolformer import *
from .regnet import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .swin_transformer import *
from .swin_transformer_v2 import *
from .vgg import *
from .vision_transformer_timm import *
from .vision_transformer_torchvision import *

__vars__ = vars()


def build_model(args):
    import torch
    from torch.hub import load_state_dict_from_url
    from termcolor import cprint
    from .. import datasets
    from ..utils.io import checkpoint_loader
    from ..utils.misc import is_main_process

    model_lib = args.model_lib.lower()
    model_name = args.model.lower()

    if 'num_classes' in args.model_kwargs.keys():
        cprint(f"Warning: Do NOT set 'num_classes' in 'args.model_kwargs'. "
               f"Now fetching the 'num_classes' registered in 'qtcls/datasets/__init__.py'.", 'light_yellow')

    try:
        num_classes = datasets.num_classes[args.dataset.lower()]
    except KeyError:
        print(f"KeyError: 'num_classes' for the dataset '{args.dataset.lower()}' is not found. "
              f"Please register your dataset's 'num_classes' in 'qtcls/datasets/__init__.py'.")
        exit(1)

    args.model_kwargs['num_classes'] = num_classes

    pretrained = not args.no_pretrain and is_main_process()

    if model_lib == 'torchvision-ex':
        try:
            model = __vars__[model_name](**args.model_kwargs)
        except KeyError:
            print(f"KeyError: Model '{model_name}' is not found.")
            exit(1)

        if pretrained:
            found_local_path = search_pretrained_from_local_paths(model_name)
            found_url = search_pretrained_from_urls(model_name)

            if found_local_path:
                state_dict = torch.load(found_local_path)
            elif found_url:
                state_dict = load_state_dict_from_url(found_url, progress=True)
                if 'model' in state_dict.keys():
                    state_dict = state_dict['model']
            else:
                raise FileNotFoundError(f"Pretrained model for '{model_name}' is not found. "
                                        f"Please register your pretrained path in 'qtcls/datasets/_pretrain_.py' "
                                        f"or set the argument '--no_pretrain'.")

            checkpoint_loader(model, state_dict, strict=False)

        return model

    if model_lib == 'timm':
        import timm
        return timm.create_model(model_name=model_name, pretrained=pretrained, **args.model_kwargs)

    raise ValueError(f"Model lib '{model_lib}' is not found.")


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
