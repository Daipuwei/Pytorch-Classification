# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 下午9:22
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : __init__.py.py
# @Software: PyCharm

from ...utils.registry_utils import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

from .mnist import MNIST