# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 下午9:28
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : bases.py
# @Software: PyCharm

"""
    这是定义基础数据集类的脚本
"""

import os
import logging
import numpy as np
from tabulate import tabulate
from termcolor import colored
from collections import OrderedDict

logger = logging.getLogger(__name__)

class Dataset(object):

    def __init__(self, train, val,category_list,logger,transform=None, mode='train',verbose=True, **kwargs):
        """
        这是抽象数据集类的初始化函数
        Args:
            train: 训练集列表，每个元素(图像路径，图像分类名称)组成的元组
            val: 验证集列表，每个元素(图像路径，图像分类名称)组成的元组
            category_list: 分类类别列表
            logger: 日志类实例
            transform: 图像转换操作，默认为None
            mode: 数据集模式，默认'train'，代表训练集，，候选值有['train','val']
            verbose: 是否可视化数据集信息标志位，，默认为True
            **kwargs: 参数字典
        """
        self._train = train
        self._val = val
        self.transform = transform
        self.mode = mode
        self.verbose = verbose
        self.logger = logger
        self.category_dict = OrderedDict(zip(category_list,np.arange(len(category_list))))

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'val':
            self.data = self.val
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | val].'.format(self.mode))

    @property
    def train(self):
        if callable(self._train):
            self._train = self._train()
        return self._train

    @property
    def val(self):
        if callable(self._val):
            self._val = self._val()
        return self._val

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        image_path_set = set()
        category_set = set()
        for info in data:
            image_path_set.add(info[0])
            category_set.add(info[1])
        return len(image_path_set), len(category_set)

    def get_num_images(self, data):
        """
        这是获取图像个数的函数
        Args:
            data: 数据列表，每个元素为(图像路径，图像分类名称)
        Returns:
        """
        return self.parse_data(data)[0]

    def get_num_categories(self, data):
        """
        这是获取分类个数的函数
        Args:
            data: 数据列表，每个元素为(图像路径，图像分类名称)
        Returns:
        """
        return self.parse_data(data)[1]

    def show_summary(self):
        """
        这是展示数据集统计信息的函数
        Returns:
        """
        pass

    def check_before_run(self, required_files):
        """
        这是解析数据集之前检查数据集是否包文件函数
        Args:
            required_files: 文件路径名称数组
        Returns:
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        # 遍历所有文件路径，判断文件是否存在，不存在则之间抛出错误
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise RuntimeError('"{}" is not found'.format(file_path))

class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def show_train(self):
        """
        这是展示训练集信息的函数
        Returns:
        """
        # 解析训练集数据规模和标签大小
        num_train_images,_ = self.parse_data(self.train)

        # 统计每个类别数据规模
        category_num_dict = OrderedDict(sorted(self.category_dict.items(), key=lambda t: t[0]))
        for _,category in self.train:
            if category not in category_num_dict.keys():
                category_num_dict[category] = 1
            else:
                category_num_dict[category] += 1

        # 初始化训练集统计信息表
        headers = ['subset', '# categories', '# images']
        csv_results = [['train', "all", num_train_images]]
        for category,num_category in category_num_dict.items():
            csv_results.append(['train', category, num_category])

        # 可视化训练集统计信息
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        self.logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))

    def show_val(self):
        """
        这是展示验证集信息的函数
        Returns:
        """
        # 解析验证集数据规模和标签大小
        num_val_images, _ = self.parse_data(self.val)

        # 统计每个类别数据规模
        category_num_dict = OrderedDict(sorted(self.category_dict.items(), key=lambda t: t[0]))
        for _, category in self.val:
            if category not in category_num_dict.keys():
                category_num_dict[category] = 1
            else:
                category_num_dict[category] += 1

        # 初始化验证集统计信息表
        headers = ['subset', '# categories', '# images']
        csv_results = [['val', "all", num_val_images]]
        for category, num_category in category_num_dict.items():
            csv_results.append(['val', category, num_category])

        # 可视化验证集统计信息
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        self.logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
