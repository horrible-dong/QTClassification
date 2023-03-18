# Copyright (c) QIU, Tian. All rights reserved.

from utils.misc import is_main_process


def getattr_case_insensitive(func):
    def wrapper(*args, **kwargs):
        import builtins as __builtin__
        ori_getattr = __builtin__.getattr

        def getattr(obj, name, default=None):
            for a in dir(obj):
                if a.lower() == name.lower():
                    return ori_getattr(obj, a, default)
            raise AttributeError(f"module '{obj.__name__}' has no attribute '{name}'")

        __builtin__.getattr = getattr
        ret = func(*args, **kwargs)
        __builtin__.getattr = ori_getattr

        return ret

    return wrapper


def main_process(func):
    def wrapper(*args, **kwargs):
        if is_main_process():
            ret = func(*args, **kwargs)
            return ret

    return wrapper
