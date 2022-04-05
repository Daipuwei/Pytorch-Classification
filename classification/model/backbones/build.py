# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 下午2:20
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是搭建backbone网络架构的函数
"""

from ...utils.registry_utils import Registry

# 初始化backbone注册类
BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`classification.config.CfgNode`
It must returns an instance of :class:`Backbone`.
"""

def build_backbone(cfg):
    """
    这是根据参数配置类搭建backbone的函数
    Args:
        cfg: 参数配置类
    Returns:
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone
