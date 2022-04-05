# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 下午2:00
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : registry_utils.py
# @Software: PyCharm

"""
    这是定义动态注册类的函数，复制于：https://github.com/JDAI-CV/fast-reid/fastreid/utils/registry.py
"""

from typing import Dict, Optional

class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        这是注册类的初始化函数
        Args:
            name (str): 注册类的名称
        Returns:
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name, obj):
        """
        这是执行注册的函数
        Args:
            name: 名称
            obj: 目标
        Returns:
        """
        assert (
                name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        # 更新名称-目标字典
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        根据目标(对象)的系统名称进行注册的函数
        Args:
            obj: 目标(对象)
        Returns:
        """
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name):
        """
        这是根据名称获取目标(对象)的函数
        Args:
            name: 名称
        Returns:
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret
