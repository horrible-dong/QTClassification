# Copyright (c) QIU Tian. All rights reserved.

import importlib
import inspect
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import cprint

from .decorators import main_process_only


def pil_loader(path, format=None):
    image = Image.open(path)
    if format is not None:
        image = image.convert(format)
    return image


# @main_process_only
def pil_saver(image, path, mode=0o777, overwrite=True):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    image.save(path)
    os.chmod(path, mode)


def cv2_loader(path, format=None):
    if format is None:  # default: BGR
        image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    elif format == 'RGB':
        image = np.array(pil_loader(path, format='RGB'))
    elif format == 'L':
        image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    elif format == 'UNCHANGED':
        image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError
    return image


# @main_process_only
def cv2_saver(image, path, mode=0o777, overwrite=True):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    cv2.imwrite(path, image)
    os.chmod(path, mode)


def json_loader(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


# @main_process_only
def json_saver(obj, path, mode=0o777, overwrite=True, **kwargs):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, **kwargs)
    os.chmod(path, mode)


def checkpoint_loader(obj, checkpoint, load_pos=None, delete_keys=(), strict=False, verbose=True):
    obj_state_dict = obj.state_dict()
    new_checkpoint = {}
    incompatible_value_shape = []
    for k, v in checkpoint.items():
        if k in obj_state_dict.keys():
            if not hasattr(v, 'shape') or (hasattr(v, 'shape') and v.shape == obj_state_dict[k].shape):
                new_checkpoint[k.replace('module.', '')] = v
            else:
                incompatible_value_shape.append(k)
    checkpoint = new_checkpoint

    for key in delete_keys:
        pop_info = checkpoint.pop(key, KeyError)
        if pop_info is KeyError:
            cprint(f"Warning: The key '{key}' to be deleted is not in the checkpoint.", 'light_yellow')

    if load_pos is not None:
        obj = getattr(obj, load_pos)

    def has_param(func, param_name: str) -> bool:
        signature = inspect.signature(func)
        return any(param.name == param_name for param in signature.parameters.values())

    if has_param(obj.load_state_dict, 'strict'):
        load_info = obj.load_state_dict(checkpoint, strict=strict)
    else:
        load_info = obj.load_state_dict(checkpoint)

    if verbose and load_info:
        if str(load_info) == '<All keys matched successfully>':
            cprint(load_info, 'light_green')
        else:
            # if len(incompatible_value_shape) > 0:
            #     cprint(f'Warning: _IncompatibleValueShape({incompatible_value_shape})', 'light_yellow')
            cprint(f'Warning: {load_info}', 'light_yellow')


@main_process_only
def checkpoint_saver(obj, save_path, mode=0o777, rename=False, overwrite=True):
    if os.path.exists(save_path):
        if not rename and not overwrite:
            raise FileExistsError
        if overwrite and rename:
            raise Exception('overwrite or rename?')

        if overwrite:
            os.remove(save_path)
        if rename:
            while os.path.exists(save_path):
                split_path = os.path.splitext(save_path)
                save_path = split_path[0] + '(1)' + split_path[1]

    torch.save(obj, save_path)
    os.chmod(save_path, mode)


def variables_loader(file_path):
    r"""
    The parameter 'file_path' supports any file system path.
    E.g. configs/_demo_.py, D:\\QTClassification\\configs\\_demo_.py, ../../other_project/cfg.py
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"file '{file_path}' does not exist.")

    dir_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    sys.path.append(dir_path)
    module_name = os.path.splitext(file_name)[0]
    module = importlib.import_module(module_name)

    variables = {}
    for name, value in inspect.getmembers(module):
        if not name.startswith('__') and not inspect.ismodule(value):
            variables[name] = value

    return variables


@main_process_only
def variables_saver(variables: dict, save_path, mode=0o777):
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'w') as f:
        for k, v in variables.items():
            f.write(f'{k} = {repr(v)}\n')
    os.chmod(save_path, mode)


@main_process_only
def log_writer(log_path, content: str, mode=0o777):
    log_exists = True if os.path.exists(log_path) else False
    with open(log_path, 'a') as f:
        f.write(content + '\n')
    if not log_exists:
        os.chmod(log_path, mode)
