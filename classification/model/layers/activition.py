# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 下午3:44
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : activition.py
# @Software: PyCharm

"""
    这是定义激活函数的脚本
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeLU(nn.Module):

    def __init__(self):
        super(GeLU,self).__init__()
        relu = nn.Hardswish

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Mish(nn.Module):

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activition(activition_type):
    """
    这是初始化激活函数层的函数
    Args:
        activition_type: 激活函数类型
    Returns:
    """
    all_activations = ["ReLU", "ReLU6", "LeakyReLU", "Swish", "Mish", "HardSwish",
                       "GeLU", "Sigmoid", "HardSigmoid", "Tanh","HardTanh"]
    # 激活函数类型不在候选激活函数中，则强制选择ReLU
    if activition_type not in all_activations:
        activition_type = "ReLU"
    # 初始化激活函数
    activition = {"ReLU": nn.ReLU(inplace=True),
                  "ReLU6": nn.ReLU6(inplace=True),
                  "LeakyReLU": nn.LeakyReLU(inplace=True),
                  "Swish": Swish(),
                  "Mish": Mish(),
                  "HardSwish": nn.Hardswish(inplace=True),
                  "GeLU": GeLU(),
                  "Sigmoid": nn.Sigmoid(),
                  "HardSigmoid": nn.Hardsigmoid(inplace=True),
                  "Tanh": nn.Tanh(),
                  "HardTanh": nn.Hardtanh(inplace=True),
                  }[activition_type]
    return activition