#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1 22:19
# @Author  : lizimo@nuist.edu.cn
# @File    : log.py
# @Description:
import logging
from logging.handlers import RotatingFileHandler
import sys
from rich.logging import RichHandler


def get_logger(name: str = "app_logger",
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        return logger
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger