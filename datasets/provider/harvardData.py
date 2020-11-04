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
from tnf_transform.img_process import random_affine, generator_affine_param, generate_affine_param
from tnf_transform.transformation import AffineTnf, affine_transform_opencv_batch
from traditional_ntg.image_util import symmetricImagePad, scale_image
from util.pytorchTcv import param2theta, theta2param, inverse_theta
from util.time_util import calculate_diff_time

'''
用作训练用，随机产生训练仿射变换参数然后输入网络进行训练。
作为dataloader的参数输入，自定义getitem得到Sample{image,theta,name}
'''
class HarvardData(Dataset):


    def __init__(self,training_image_path,paper_affine_generator = False,transform=None,cache_images = False,use_cuda = False):
        '''
        :param training_image_path:
        :param output_size:
        :param transform:
        :param cache_images:    如果数据量不是特别大可以缓存到内存里面加快读取速度
        :param use_cuda:
        '''
        self.use_cuda = use_cuda
        self.cache_images = cache_images
        self.paper_affine_generator = paper_affine_generator
        # read image file
        self.training_image_path = training_image_path
        self.train_data = sorted(os.listdir(self.training_image_path))
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

        print('center:',"120,120")
        # center = (256,256)
        center = (120,120)
        # print('center:',"0,0")
        # center = (0,0)
        small = True
        if small:
            # 注意，这里x和y方向的偏移像素的话要考虑原图的大小，比如我的原图是240*240，ntg原论文是512*512，像素偏移的太大对结果影响很大
            self.theta = generate_affine_param(scale=1.1, degree=10, translate_x=-10, translate_y=10, center=center)
        else:
            self.theta = generate_affine_param(scale=1.25, degree=30, translate_x=-20, translate_y=20, center=center)
        # 将opencv的参数转换为pytorch的参数
        self.theta = torch.from_numpy(self.theta.astype(np.float32)).expand(1,2,3)
        self.theta = param2theta(self.theta, 512, 512, use_cuda=self.use_cuda)[0]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        image_name = self.train_data[idx]

        if self.cache_images:
            image = self.imgs[idx]
        else:
            image_path = os.path.join(self.training_image_path, image_name)

            array_struct = scio.loadmat(image_path)
            #array_data = array_struct['ms_image_denoised']  # harvard数据
            array_data = array_struct['cave_mat']  # cave_mat数据

        image = np.ascontiguousarray(array_data, dtype=np.float32)  # uint8 to float32
        image = torch.from_numpy(image)

        # image (h,w,channel)
        sample = {'image': image, 'theta': self.theta, 'name': image_name,'raw_image':image.clone()}

        # print(self.transform is None)
        if self.transform:
            sample = self.transform(sample)

        return sample

class HarvardDataPair(object):

    def __init__(self,single_channel=False, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.6):
        self.single_channel = single_channel
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)
        self.geometricTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)
        self.rawTnf = AffineTnf(512,512,use_cuda=self.use_cuda)

    def __call__(self, batch):
        # 由dataloader返回的数据image为(batch,h,w,channel)
        image_batch, theta_batch,image_name,raw_image_batch = batch['image'], batch['theta'],batch['name'],batch['raw_image']

        batch_size,h,w,channel = image_batch.shape

        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        # 这里batch设置为1，channel为31，所以调换顺序得到(channel,1,h,w)，符合常规处理流程
        image_batch = image_batch.transpose(2,3).transpose(1,2).transpose(0,1)

        # 因为计算的时候是三通道，所以这里使用叠加得到(channel,3,h,w)
        if not self.single_channel:
            image_batch = torch.cat((image_batch,image_batch,image_batch),1)

        self.padding_factor = 1.0
        self.crop_factor = 1.0

        theta_batch = theta_batch.expand(image_batch.shape[0],2,3)

        # 为较大的采样区域生成对称填充图像
        # image_batch = symmetricImagePad(image_batch, self.padding_factor,use_cuda=self.use_cuda)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given

        warped_image_batch = self.geometricTnf(image_batch, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        raw_source_image_batch = image_batch
        raw_target_image_batch = self.rawTnf(image_batch,theta_batch,self.padding_factor,self.crop_factor)

        spec_channel = torch.tensor([16])
        if self.use_cuda:
            image_batch = image_batch.cuda()
            warped_image_batch = warped_image_batch.cuda()
            theta_batch = theta_batch.cuda()
            spec_channel = spec_channel.cuda()

        b, c, h, w = warped_image_batch.size()

        # warped_image_batch = torch.index_select(warped_image_batch,0,spec_channel).expand(b,c,h,w)
        raw_target_image_batch = torch.index_select(raw_target_image_batch,0,spec_channel).expand(b,c,512,512)

        # return {'source_image': cropped_image_batch, 'target_image': warped_image_batch,
        #         'raw_source_image_batch': raw_source_image_batch, 'raw_target_image_batch': raw_target_image_batch,
        #         'theta_GT': theta_batch,'name':image_name}

        theta_batch = inverse_theta(theta_batch,use_cuda=True)

        cropped_image_batch = torch.index_select(cropped_image_batch, 0, spec_channel).expand(b, c, h, w)

        return {'source_image': warped_image_batch, 'target_image': cropped_image_batch,
                'raw_source_image_batch': raw_source_image_batch, 'raw_target_image_batch': raw_target_image_batch,
                'theta_GT': theta_batch,'name':image_name}


class HarvardRawDataPair(object):

    def __init__(self,single_channel=False, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.6):
        self.single_channel = single_channel
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)
        self.geometricTnf = AffineTnf(self.out_h, self.out_w,use_cuda=self.use_cuda)

    def __call__(self, batch):
        # 由dataloader返回的数据image为(batch,h,w,channel)
        image_batch, theta_batch,image_name,raw_image_batch = batch['image'], batch['theta'],batch['name'],batch['raw_image']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        # 这里batch设置为1，channel为31，所以调换顺序得到(channel,1,h,w)，符合常规处理流程
        image_batch = image_batch.transpose(2,3).transpose(1,2).transpose(0,1)

        # 因为计算的时候是三通道，所以这里使用叠加得到(channel,3,h,w)
        if not self.single_channel:
            image_batch = torch.cat((image_batch,image_batch,image_batch),1)

        self.padding_factor = 1.0
        self.crop_factor = 1.0

        theta_batch = theta_batch.expand(image_batch.shape[0],2,3)

        # 为较大的采样区域生成对称填充图像
        # image_batch = symmetricImagePad(image_batch, self.padding_factor,use_cuda=self.use_cuda)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given

        warped_image_batch = self.geometricTnf(image_batch, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        spec_channel = torch.tensor([16])
        if self.use_cuda:
            spec_channel = spec_channel.cuda()

        b, c, h, w = warped_image_batch.size()

        warped_image_batch = torch.index_select(warped_image_batch,0,spec_channel).expand(b,c,h,w)


        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}