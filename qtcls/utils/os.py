# Copyright (c) QIU Tian. All rights reserved.

import os
import shutil

from qtcls.utils.decorators import main_process_only


# Do NOT decorate this function with @main_process_only.
def chmod(path, mode, **kwargs):
    try:
        os.chmod(path, mode, **kwargs)
    except PermissionError:
        pass
        # cprint(f"PermissionError: 'chmod' operation not permitted: '{path}'", 'light_yellow')


@main_process_only
def mkdir(path, mode=0o777):
    os.mkdir(path)
    chmod(path, mode)


@main_process_only
def makedirs(path, mode=0o777, exist_ok=False):
    def _makedirs_impl(path, mode, exist_ok):
        head, tail = os.path.split(path)
        if not tail:
            head, tail = os.path.split(head)
        if head and tail and not os.path.exists(head):
            try:
                _makedirs_impl(head, mode, exist_ok)
            except FileExistsError:
                pass
            cdir = os.path.curdir
            if isinstance(tail, bytes):
                cdir = bytes(os.path.curdir, 'ASCII')
            if tail == cdir:
                return
        try:
            os.mkdir(path)
            chmod(path, mode)
        except OSError:
            if not exist_ok or not os.path.isdir(path):
                raise

    _makedirs_impl(path, mode, exist_ok)


@main_process_only
def symlink(src_path, symlink_path, mode=0o777):
    os.symlink(src_path, symlink_path)
    chmod(symlink_path, mode)


@main_process_only
def rmtree(path, not_exist_ok=False):
    if not_exist_ok and not os.path.exists(path):
        return
    shutil.rmtree(path)
