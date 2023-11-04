# Copyright (c) QIU Tian. All rights reserved.

import os


def mkdir(path, mode=0o777):
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path, mode)


def makedirs(path, mode=0o777, exist_ok=False):  # CANNOT be decorated by '@main_process_only'
    head, tail = os.path.split(path)
    if not tail:
        head, tail = os.path.split(head)
    if head and tail and not os.path.exists(head):
        try:
            makedirs(head, exist_ok=exist_ok)
        except FileExistsError:
            pass
        cdir = os.path.curdir
        if isinstance(tail, bytes):
            cdir = bytes(os.path.curdir, 'ASCII')
        if tail == cdir:
            return
    try:
        mkdir(path, mode)
    except OSError:
        if not exist_ok or not os.path.isdir(path):
            raise


def symlink(src_path, symlink_path, mode=0o777):
    os.symlink(src_path, symlink_path)
    os.chmod(symlink_path, mode)

