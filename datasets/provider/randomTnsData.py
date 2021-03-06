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

from tnf_transform.img_process import random_affine, generator_affine_param
from tnf_transform.transformation import AffineTnf
from traditional_ntg.image_util import symmetricImagePad
from util.pytorchTcv import inverse_theta
from util.time_util import calculate_diff_time

'''
用作训练用，随机产生训练仿射变换参数然后输入网络进行训练。
作为dataloader的参数输入，自定义getitem得到Sample{image,theta,name}
'''
class RandomTnsData(Dataset):


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
            image = cv2.imread(image_path)  # shape [h,w,c]

        if image.shape[0]!= self.out_h or image.shape[1] != self.out_w:
            image = cv2.resize(image, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
        image = torch.from_numpy(image)

        if self.paper_affine_generator:
            theta = generator_affine_param()
        else:
            theta = random_affine()

        theta = torch.from_numpy(theta.astype(np.float32))

        sample = {'image': image, 'theta': theta, 'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        # elpased = calculate_diff_time(total_start_time)
        # print('getitem时间:',elpased)     # 0.011s

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
        image_batch = symmetricImagePad(image_batch, self.padding_factor,use_cuda=self.use_cuda)

        # indices_R = torch.tensor([choice(self.channel_choicelist)])
        # indices_G = torch.tensor([choice(self.channel_choicelist)])

        indices_R = torch.tensor([0])
        indices_G = torch.tensor([2])

        if self.use_cuda:
            indices_R = indices_R.cuda()
            indices_G = indices_G.cuda()

        image_batch_R = torch.index_select(image_batch, 1, indices_R)
        image_batch_G = torch.index_select(image_batch, 1, indices_G)

        image_batch_R = torch.cat((image_batch_R,image_batch_R,image_batch_R),1)
        image_batch_G = torch.cat((image_batch_G,image_batch_G,image_batch_G),1)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(image_batch_R, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given
        # 获取裁剪变换的图像
        warped_image_batch = self.geometricTnf(image_batch_G, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}

        # theta_batch = inverse_theta(theta_batch,use_cuda=True)
        #
        # return {'source_image': warped_image_batch, 'target_image': cropped_image_batch, 'theta_GT': theta_batch,
        #         'name':image_name}

class RandomTnsPairSingleChannelTest(object):

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
        image_batch = symmetricImagePad(image_batch, self.padding_factor,use_cuda=self.use_cuda)

        # indices_R = torch.tensor([choice(self.channel_choicelist)])
        # indices_G = torch.tensor([choice(self.channel_choicelist)])

        indices_R = torch.tensor([0])
        indices_G = torch.tensor([2])
        #
        if self.use_cuda:
            indices_R = indices_R.cuda()
            indices_G = indices_G.cuda()

        image_batch_R = torch.index_select(image_batch, 1, indices_R)
        image_batch_G = torch.index_select(image_batch, 1, indices_G)

        image_batch_R = torch.cat((image_batch_R,image_batch_R,image_batch_R),1)
        image_batch_G = torch.cat((image_batch_G,image_batch_G,image_batch_G),1)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(image_batch_R, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given
        # 获取裁剪变换的图像
        warped_image_batch = self.geometricTnf(image_batch_G, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}