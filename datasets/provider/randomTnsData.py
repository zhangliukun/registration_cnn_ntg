import os

from skimage import io
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import time
import cv2

from random import choice
from pathlib import Path

from tqdm import tqdm

from tnf_transform.img_process import random_affine
from tnf_transform.transformation import AffineTnf
from traditional_ntg.image_util import symmetricImagePad
from util.time_util import calculate_diff_time

'''
用作训练用，随机产生训练仿射变换参数然后输入网络进行训练。
作为dataloader的参数输入，自定义getitem得到Sample{image,theta,name}
'''
class RandomTnsData(Dataset):


    def __init__(self,training_image_path,output_size=(480,640),transform=None,cache_images = False,use_cuda = True):
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
            image = cv2.imread(image_path)  # shape [h,w,c]

        if image.shape[0]!= self.out_h or image.shape[1] != self.out_w:
            image = cv2.resize(image, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
        image = torch.from_numpy(image)

        theta = random_affine()
        theta = torch.from_numpy(theta[0:2].astype(np.float32))

        sample = {'image': image, 'theta': theta, 'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        # elpased = calculate_diff_time(total_start_time)
        # print('getitem时间:',elpased)     # 0.011s

        return sample

    def __init1__(self,training_image_path,output_size=(480,640),transform=None,random_t=0.2,random_s=0.2,
                 random_alpha = 1/4,random_sample=True,use_cuda = True):
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.out_h, self.out_w = output_size
        self.use_cuda = use_cuda
        # read image file
        self.training_image_path = training_image_path
        self.train_data = os.listdir(self.training_image_path)
        # copy arguments
        self.transform = transform
        self.affineTnf = AffineTnf(self.out_h,self.out_w,use_cuda=False)


    def __getitem1__(self, idx):
        total_start_time = time.time()
        start_time = time.time()

        image_name = self.train_data[idx]
        image_path = os.path.join(self.training_image_path, image_name)
        image = io.imread(image_path)

        # elpased = calculate_diff_time(start_time)
        # print('读入一张image：',elpased) # 0.01s

        #start_time = time.time()

        alpha = (torch.rand(1) - 0.5) * 2 * np.pi * self.random_alpha
        alpha = alpha.numpy()
        theta = torch.rand(6).numpy()

        theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * self.random_t
        theta[0] = (1 + (theta[0] - 0.5) * 2 * self.random_s) * np.cos(alpha)
        theta[1] = (1 + (theta[1] - 0.5) * 2 * self.random_s) * (-np.sin(alpha))
        theta[3] = (1 + (theta[3] - 0.5) * 2 * self.random_s) * np.sin(alpha)
        theta[4] = (1 + (theta[4] - 0.5) * 2 * self.random_s) * np.cos(alpha)
        theta = theta.reshape(2, 3)

        # elpased = calculate_diff_time(start_time)
        # print('计算随机变换参数：',elpased)  # 0.0004s

        # # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))

        #start_time = time.time()
        # # permute order of image to CHW
        try:
            image = image.transpose(1, 2).transpose(0, 1)
        except Exception:
            # 图片中有些图片只是单通道，如[h,w]
            one = image.unsqueeze(0)
            image = torch.cat((one, one, one), 0)

        # elpased = calculate_diff_time(start_time)
        # print('transpose:',elpased)  # 0.00010s

        # # 时间有点长，放入dataloader的GPU处理中
        # start_time = time.time()
        # # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0] != self.out_h or image.size()[1] != self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0), requires_grad=False)).data.squeeze(0)
        #
        # elpased = calculate_diff_time(start_time)
        # print('缩放图片:',elpased)  # 0.18s



        sample = {'image': image, 'theta': theta, 'name': image_name}

        # start_time = time.time()
        #
        if self.transform:
            image = self.transform(sample)
        #
        # elpased = calculate_diff_time(start_time)
        # print('归一化图片:',elpased)     # 0.0006s

        # elpased = calculate_diff_time(total_start_time)
        # print('getitem时间:',elpased)     # 0.036s

        return sample

'''
使用仿射变换参数生成图片对
返回{"source_image,traget_image,theta_GT,name"}
'''
class RandomTnsPair(object):

    def __init__(self, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.6):
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.channel_choicelist = [0,1,2]
        self.rescalingTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)
        self.geometricTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)

    def __call__(self, batch):
        image_batch, theta_batch,image_name = batch['image'], batch['theta'],batch['name']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        b, c, h, w = image_batch.size()

        # 为较大的采样区域生成对称填充图像
        image_batch = symmetricImagePad(image_batch, self.padding_factor)

        # convert to variables 其中Tensor是原始数据，并不知道梯度计算等问题，
        # Variable里面有data，grad和grad_fn，其中data就是Tensor
        # image_batch = Variable(image_batch, requires_grad=False)
        # theta_batch = Variable(theta_batch, requires_grad=False)



        indices_R = torch.tensor([choice(self.channel_choicelist)])
        indices_G = torch.tensor([choice(self.channel_choicelist)])

        if self.use_cuda:
            indices_R = indices_R.cuda()
            indices_G = indices_G.cuda()

        image_batch_R = torch.index_select(image_batch, 1, indices_R)
        image_batch_G = torch.index_select(image_batch, 1, indices_G)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(image_batch_R, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given
        # 获取裁剪变换的图像
        warped_image_batch = self.geometricTnf(image_batch_G, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}