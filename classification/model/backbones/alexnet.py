# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 下午2:41
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : alexnet.py
# @Software: PyCharm

"""
    这是定义AlexNet的脚本
"""

import os

import numpy as np
import torch
import logging
from torch import nn
from torchvision.models import AlexNet

from .build import BACKBONE_REGISTRY
from ..layers.activition import get_activition

logger = logging.getLogger(__name__)
model_urls = {"ReLU":"https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",}

class CONV(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,activition_type='ReLU'):
        """
        这是卷积+激活函数模块的初始化函数
        Args:
            in_channels: 卷积层输入维度
            out_channels: 卷积层输出维度
            kernel_size: 内核大小，默认为3
            stride: 步长，默认为1
            padding: 填充像素大小，默认为1
            activition_type: 激活函数类型,默认为“ReLU”
        """
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                kernel_size=kernel_size,stride=stride,padding=padding)
        self.activition = get_activition(activition_type=activition_type)

    def forward(self,x):
        """
        卷积+激活函数模块的前向传播函数
        Args:
            x: 输入张量
        Returns:
        """
        output = self.conv2d(x)
        output = self.activition(output)
        return output

class CONV_MaxPool(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activition_type='ReLU'):
        """
        这是卷积+激活函数+最大池化模块的初始化函数
        Args:
            in_channels: 卷积层输入维度
            out_channels: 卷积层输出维度
            kernel_size: 内核大小，默认为3
            stride: 步长，默认为1
            padding: 填充像素大小，默认为0
            activition_type: 激活函数类型,默认为“ReLU”
        """
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.activition = get_activition(activition_type=activition_type)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        """
        卷积+激活函数+最大池化模块的前向传播函数
        Args:
            x: 输入张量
        Returns:
        """
        x = self.conv2d(x)
        x = self.activition(x)
        x = self.max_pool(x)
        return x

class Dense(nn.Module):

    def __init__(self,in_channels, out_channels,activition_type="ReLU"):
        """
        这是全连接层+激活函数模块的初始化函数
        Args:
            in_channels: 卷积层输入维度
            out_channels: 卷积层输出维度
            activition_type: 激活函数类型,默认为“ReLU”
        """
        self.linear = nn.Linear(in_features=in_channels,out_features=out_channels)
        self.activition = get_activition(activition_type)

    def forward(self,x):
        """
        这是全连接层+激活函数模块的前向传播函数
        Args:
            x: 输入张量
        Returns:
        """
        x = self.linear(x)
        x = self.activition(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, dropout_prob=0.5,activition_type='ReLU',is_eval=False):
        """
        这是AlexNet的初始化函数
        Args:
            num_classes: 分类个数，默认为1000
            dropout_prob: Dropout层概率值,默认为0.5
            activition_type: 激活函数类型，默认为"ReLU"
            is_eval: 是否为评估阶段，默认为False
        """
        super().__init__()
        self.features = nn.Sequential(
            CONV_MaxPool(3, 64, kernel_size=11, stride=4, padding=2, activition_typ=activition_type),
            CONV_MaxPool(64, 192, kernel_size=5, padding=2, activition_type=activition_type),
            CONV(192, 384, kernel_size=3, padding=1, activition_type=activition_type),
            CONV(384, 256, kernel_size=3, padding=1, activition_type=activition_type),
            CONV_MaxPool(256, 256, kernel_size=3, padding=1, activition_type=activition_type),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            Dense(256 * 6 * 6, 4096,activition_type=activition_type),
            nn.Dropout(p=dropout_prob),
            Dense(4096,4096,activition_type=activition_type),
            nn.Linear(4096, num_classes),
        )
        self.is_eval = is_eval
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        batch_size,c,h,w = x.size()
        x = x.view((-1,c*h*w))
        x = self.classifier(x)
        if self.is_eval:        # 测试阶段加入softmax层
            x = self.softmax(x)
        return x

def init_pretrained_weights(key):
    """
    这是根据关键词获取指定Alexnet预训练权重网址的函数
    Args:
        key: 关键词
    Returns:
    """

    import os
    import errno
    import gdown

    def _get_torch_home():
        """
        这是生成pytorch预训练权重目录地址的函数
        Returns:
        """
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home
    
    # 初始化预训练权重保存目录地址
    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    
    # 下载并加载预训练权重字典
    state_dict = {}
    if key in model_urls.keys():
        filename = model_urls[key].split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):         # 模型文件不存在，则自动下载模型
            logger.info("Pretrain model don't exist, downloading from {0}".format(model_urls[key]))
            gdown.download(model_urls[key], cached_file, quiet=False)
        logger.info(f"Loading pretrained model from {cached_file}")
        state_dict = torch.load(cached_file, map_location=torch.device('cpu'))
    else:
        logger.info("No pretrained model found!")
    return state_dict

@BACKBONE_REGISTRY.register()
def build_alexnet(cfg):
    """
    这是搭建AlexNet的函数
    Args:
        cfg: 参数配置类
    Returns:
    """
    # 初始化AlexNet相关参数
    num_classes = cfg.MODEL.BACKBONE.NUM_CLASSES
    drop_prob = cfg.MODEL.BACKBONE.DROP_PROB
    activition_type = cfg.MODEL.BACKBONE.ACTIVITION_TYPE
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_weigth_path = os.path.abspath(cfg.MODEL.BACKBONE.PRETRAIN_WEIGHT_PATH)

    # 初始化AlexNet
    alexnet = AlexNet(num_classes=num_classes,dropout_prob=drop_prob,activition_type=activition_type)
    if pretrain:     # 可以加载预训练权重
        if pretrain_weigth_path:        # 预训练权重文件存在，加载预训练权重字典
            try:
                pretrain_state_dict = torch.load(pretrain_weigth_path, map_location=torch.device('cpu'))
                logger.info("Loading AlexNet pretrained model from {0}".format(pretrain_weigth_path))
            except FileNotFoundError as e:
                logger.debug('{0} is not found! Please check this path.'.format(pretrain_weigth_path))
            except KeyError as e:
                logger.debug("State dict keys error! Please check the state dict.")
        else:                           # 预训练权重文件不存在
            if activition_type == 'ReLU':
                key = 'ReLU'
            else:
                key = ''
            # 自动下载预训练权重文件并初始化预训练权重字典
            pretrain_state_dict = init_pretrained_weights(key)
        # 加载预训练权重
        if len(pretrain_state_dict) != 0:
            model_dict = alexnet.state_dict()
            # 以此遍历所有参数权重
            for model_key,pretrain_model_key in zip(list(model_dict.keys()),list(pretrain_state_dict.keys())):
                # 参数名称可能存在包含关系，并且权重维度一致则加载
                if pretrain_model_key in model_key:
                    if np.shape(model_dict[model_key]) == pretrain_state_dict[pretrain_model_key]:
                        model_dict[model_key] = pretrain_state_dict[pretrain_model_key]
            # 加载预训练模型权重
            alexnet.load_state_dict(model_dict, strict=False)

    return alexnet