# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 下午4:57
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : mnist.py
# @Software: PyCharm

import os
import cv2
import gzip
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

from .bases import ImageDataset
from classification.data.datasets import DATASET_REGISTRY

def save_single_image(gray_image,image_path):
    """
    这是保存MNIST图片的函数
    Args:
        gray_image: MNIST灰度图像，np.array格式
        image_path: 图像路径
    Returns:
    """
    rgb_image = np.expand_dims(gray_image,axis=-1)
    rgb_image = np.concatenate([rgb_image,rgb_image,rgb_image],axis=-1)
    cv2.imwrite(image_path,rgb_image)

def save_batch_images(batch_gray_images,batch_image_paths):
    """
    这是批量保存MNIST图像的函数
    Args:
        batch_gray_images: 批量灰度图像数组，np.array格式
        batch_image_paths: 批量图像路径数组
    Returns:
    """
    size = len(batch_image_paths)
    for i in tqdm(np.arange(size)):
        save_single_image(batch_gray_images[i],batch_image_paths[i])

@DATASET_REGISTRY.register()
class MNIST(ImageDataset):

    dataset_name = "mnist"

    def __init__(self,logger,root="datasets",**kwargs):
        """
        这是MNIST数据集类的初始化函数
        Args:
            logger: 日志类实例
            root: 数据集存放根目录
            **kwargs:
        """
        # 初始化相关参数
        self.logger = logger
        self.root = os.path.join(os.getcwd(),root)
        self.dataset_dir = os.path.join(self.root,self.dataset_name)

        # 初始化相关文件夹路径
        self.train_images_gz_path = os.path.join(self.dataset_dir, "train-images-idx3-ubyte.gz")
        self.train_labels_gz_path = os.path.join(self.dataset_dir, "train-labels-idx1-ubyte.gz")
        self.val_images_gz_path = os.path.join(self.dataset_dir, "t10k-images-idx3-ubyte.gz")
        self.val_labels_gz_path = os.path.join(self.dataset_dir, "t10k-labels-idx1-ubyte.gz")
        self.train_dir = os.path.join(self.dataset_dir,'train')
        self.val_dir = os.path.join(self.dataset_dir,'val')

        # 解压MNIST数据集压缩包，生成图像文件夹
        self.preprocess_init()

        # 导入训练和验证数据集
        train = self.process_dir(self.train_dir)
        val = self.process_dir(self.val_dir)
        category_list = [str(i) for i in np.arange(10)]

        super(MNIST,self).__init__(train,val,category_list,logger,**kwargs)

    def preprocess_init(self):
        """
        这是初始化MNIST数据集的函数
        Returns:
        """
        if os.path.exists(self.train_dir) and os.path.exists(self.val_dir):
            return

        # 解压MNIST数据集压缩包，生成图像文件夹
        self.gz2image(self.train_images_gz_path, self.train_labels_gz_path, 'train')
        self.gz2image(self.val_images_gz_path, self.val_labels_gz_path, 'val')

    def gz2image(self,image_gz_path,label_gz_path,mode='train'):
        """
        这是将MNIST数据集的gz压缩包转换为图像的函数
        Args:
            image_gz_path: 图像集gz文件路径
            label_gz_path: 标签集gz文件路径
            mode: 数据集类型，默认为‘train’，代表训练集，候选值为['train','val']
        Returns:
        """
        # 读取图像标签
        with gzip.open(label_gz_path, 'rb') as f:
            y_train = np.frombuffer(f.read(), np.uint8, offset=8)

        # 读取图像
        with gzip.open(image_gz_path, 'rb') as f:
            x_train = np.frombuffer(f.read(), np.uint8, offset=16)
            x_train = np.reshape(x_train,(len(y_train), 28, 28))

        # 初始化图像地址
        image_dir = os.path.join(self.dataset_dir,mode)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_paths = []
        for i,(x,y) in enumerate(zip(x_train,y_train)):
            label_dir = os.path.join(image_dir,str(y))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            image_paths.append(os.path.join(label_dir,"{0:06d}.png".format(i)))
        image_paths = np.array(image_paths)

        # 多线程生成图片
        size = len(image_paths)
        batch_size = size // (cpu_count()-1)
        pool = Pool(processes=cpu_count()-1)
        self.logger.info("start saving {} mnist images".format(mode))
        #print("start saving {} mnist images".format(mode))
        for start in np.arange(0,size,batch_size):
            end = int(np.min([start+batch_size,size]))
            pool.apply_async(save_batch_images,args=(x_train[start:end,:,:],
                                                     image_paths[start:end]))
        pool.close()
        pool.join()
        self.logger.info("finish saving {} mnist images".format(mode))
        #print("finish saving {} mnist images".format(mode))

    def process_dir(self,image_dir):
        """
        这是导入文件夹中的图像及其标签的函数
        Args:
            image_dir: 图像文件夹路径
        Returns:
        """
        data = []
        for label in os.listdir(image_dir):
            label_dir = os.path.join(image_dir,label)                       # 标签目录
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir,image_name)             # 图像路径
                data.append((image_path,label))
        data = np.array(data)
        return data