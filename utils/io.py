# Copyright (c) QIU, Tian. All rights reserved.

import os

import cv2
import imageio
import numpy as np
import torch
import ujson as json
from PIL import Image

from utils.misc import is_main_process, has_param


def pil_loader(path, format=None):
    image = Image.open(path)
    if format is not None:
        image = image.convert(format)
    return image


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
    elif format == "RGB":
        image = np.array(pil_loader(path, format="RGB"))
    elif format == "L":
        image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    elif format == "UNCHANGED":
        image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError
    return image


def cv2_saver(image, path, mode=0o777, overwrite=True):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    cv2.imwrite(path, image)
    os.chmod(path, mode)


def json_loader(path):
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def json_saver(obj, path, mode=0o777, overwrite=True, **kwargs):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)
    os.chmod(path, mode)


def checkpoint_loader(obj, checkpoint, load_pos=None, delete_keys=(), strict=False, verbose=True):
    obj_state_dict = obj.state_dict()
    new_checkpoint = {}
    incompatible_value_shape = []
    for k, v in checkpoint.items():
        if k in obj_state_dict.keys():
            if not hasattr(v, 'shape') or (hasattr(v, 'shape') and v.shape == obj_state_dict[k].shape):
                new_checkpoint[k.replace("module.", "")] = v
            else:
                incompatible_value_shape.append(k)
    checkpoint = new_checkpoint

    for key in delete_keys:
        if checkpoint.get(key) is not None:
            del checkpoint[key]

    if load_pos is not None:
        obj = getattr(obj, load_pos)

    if has_param(obj.load_state_dict, 'strict'):
        info = obj.load_state_dict(checkpoint, strict=strict)
    else:
        info = obj.load_state_dict(checkpoint)

    if verbose:
        from termcolor import cprint
        cprint(str(info) + '\n', 'red')
        # cprint(f"_IncompatibleValueShape({incompatible_value_shape})", 'red')


def checkpoint_saver(obj, save_path, mode=0o777, rename=False, overwrite=True, save_on_master=True):
    if is_main_process():
        if os.path.exists(save_path):
            if not rename and not overwrite:
                raise FileExistsError
            if rename:
                while os.path.exists(save_path):
                    split_path = os.path.splitext(save_path)
                    save_path = split_path[0] + "(1)" + split_path[1]

    if save_on_master:
        if is_main_process():
            torch.save(obj, save_path)
            os.chmod(save_path, mode)
    else:
        torch.save(obj, save_path)
        os.chmod(save_path, mode)


def images_to_gif(image_dir, gif_path, duration=None, fps=None, loop=0):
    """
    duration: interval between frames (seconds)
    fps: frames per second
    loop: play times
    """
    if fps:
        duration = 1 / fps
    image_paths = os.listdir(image_dir)
    images = [imageio.imread(os.path.join(image_dir, image_path)) for image_path in image_paths]
    imageio.mimsave(gif_path, images, "gif", duration=duration, loop=loop)
