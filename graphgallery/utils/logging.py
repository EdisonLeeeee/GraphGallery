# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os


def set_logger(name='model', filepath=None, level='INFO'):
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(level))
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filepath is not None:
        if os.path.dirname(filepath) is not '':
            if not os.path.isdir(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
        file_handle = logging.FileHandler(filename=filepath, mode="a")
        file_handle.set_name("file")
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    return logger


def get_logger(name):
    return logging.getLogger(name)
