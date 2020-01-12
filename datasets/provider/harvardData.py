import os

from skimage import io
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import time
import cv2
import scipy.io as scio

from random import choice
from pathlib import Path

from tqdm import tqdm

from ntg_pytorch.register_pyramid import ScaleTnf
from tnf_transform.img_process import random_affine, generator_affine_param, generate_affine_param, normalize_image, \
    normalize_image_simple
from tnf_transform.transformation import AffineTnf
from traditional_ntg.image_util import symmetricImagePad, scale_image
from util.pytorchTcv import param2theta
from util.time_util import calculate_diff_time

'''
用作训练用，随机产生训练仿射变换参数然后输入网络进行训练。
作为dataloader的参数输入，自定义getitem得到Sample{image,theta,name}
'''
class HarvardData(Dataset):


    def __init__(self,training_image_path,output_size=(480,640),paper_affine_generator = False,transform=None,cache_images = False,use_cuda = True):
        '''
        :param training_image_path:
        :param output_size:
        :param transform:
        :param cache_images:    如果数据量不是特别大可以缓存到内存里面加快读取速度
        :param use_cuda:
        '''
        self.out_h, self.out_w = output_size
        self.use_cuda = use_cuda
        self.cache_images = cache_images
        self.paper_affine_generator = paper_affine_generator
        # read image file
        self.training_image_path = training_image_path
        self.train_data = os.listdir(self.training_image_path)
        self.image_count = len(self.train_data)
        # bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        # nb = bi[-1] + 1  # number of batches
        self.imgs = [None]*self.image_count
        self.transform = transform
        self.control_size = 1000000000

        # cache images into memory for faster training(~5GB)
        if self.cache_images:
            for i in tqdm(range(min(self.image_count,self.control_size)),desc='Reading images'): # 最多10k张图片
                image_name = self.train_data[i]
                image_path = os.path.join(self.training_image_path, image_name)
                image = cv2.imread(image_path)  # shape [h,w,c] BGR
                assert image is not None, 'Image Not Found' + image_path
                self.imgs[i] = image

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        #total_start_time = time.time()

        image_name = self.train_data[idx]

        if self.cache_images:
            image = self.imgs[idx]
        else:
            image_path = os.path.join(self.training_image_path, image_name)

            array_struct = scio.loadmat(image_path)
            array_data = array_struct['ms_image_denoised']

        image = np.ascontiguousarray(array_data, dtype=np.float32)  # uint8 to float32

        # IMIN = np.min(image)
        # IMAX = np.max(image)

        image = torch.from_numpy(image)

        # image = scale_image(image,IMIN,IMAX)

        small = True
        if small:
            theta = generate_affine_param(scale=1.1,degree=10,translate_x=-10,translate_y=10)
        else:
            theta = generate_affine_param(scale=1.25,degree=30,translate_x=-20,translate_y=20)

        theta = torch.from_numpy(theta.astype(np.float32))
        theta = theta.expand(image.shape[-1],2,3)

        sample = {'image': image, 'theta': theta, 'name': image_name}

        # print(self.transform is None)
        if self.transform:
            sample = self.transform(sample)

        return sample

class HarvardDataPair(object):

    def __init__(self, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.6):
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)
        self.geometricTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)

    def __call__(self, batch):
        image_batch, theta_batch,image_name = batch['image'], batch['theta'],batch['name']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        image_batch = image_batch.transpose(2,3).transpose(1,2).transpose(0,1)

        IMAX = torch.max(image_batch.view(image_batch.shape[0], 1, -1), 2)[0].unsqueeze(2).unsqueeze(2)
        IMIN = torch.min(image_batch.view(image_batch.shape[0], 1, -1), 2)[0].unsqueeze(2).unsqueeze(2)
        image_batch = scale_image(image_batch, IMIN, IMAX)

        image_batch = normalize_image_simple(image_batch)

        image_batch = torch.cat((image_batch,image_batch,image_batch),1)

        theta_batch = theta_batch.squeeze(0)

        theta_batch = param2theta(theta_batch,240,240,use_cuda=self.use_cuda)

        b, c, h, w = image_batch.size()

        # 为较大的采样区域生成对称填充图像
        image_batch = symmetricImagePad(image_batch, self.padding_factor,use_cuda=self.use_cuda)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given
        # 获取裁剪变换的图像
        warped_image_batch = self.geometricTnf(image_batch, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}