# Copyright (c) QIU, Tian. All rights reserved.

import torch.distributed as dist

from .misc import is_main_process, is_dist_avail_and_initialized


def getattr_case_insensitive(func):
    def wrapper(*args, **kwargs):
        import builtins as __builtin__
        ori_getattr = __builtin__.getattr

        def getattr(obj, name, default=None):
            for a in dir(obj):
                if a.lower() == name.lower():
                    return ori_getattr(obj, a, default)
            raise AttributeError

        __builtin__.getattr = getattr
        ret = func(*args, **kwargs)
        __builtin__.getattr = ori_getattr

        return ret

    return wrapper


def main_process_only(func):
    def wrapper(*args, **kwargs):
        if is_main_process():
            ret = func(*args, **kwargs)
            if is_dist_avail_and_initialized():
                dist.barrier()
            return ret
        else:
            dist.barrier()

    return wrapper
