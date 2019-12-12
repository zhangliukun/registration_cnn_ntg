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
from util.csv_opeartor import read_csv_file
from util.time_util import calculate_diff_time

'''
用作训练用，随机产生训练仿射变换参数然后输入网络进行训练。
作为dataloader的参数输入，自定义getitem得到Sample{image,theta,name}
'''
class NirRgbData(Dataset):


    def __init__(self,nir_path,rgb_path,label_path,output_size=(480,640),paper_affine_generator = False,transform=None,use_cuda = True):
        '''
        :param training_image_path:
        :param output_size:
        :param transform:
        :param cache_images:    如果数据量不是特别大可以缓存到内存里面加快读取速度
        :param use_cuda:
        '''
        self.out_h, self.out_w = output_size
        self.use_cuda = use_cuda
        self.paper_affine_generator = paper_affine_generator
        # read image file
        self.nir_image_path = nir_path
        self.rgb_image_path = rgb_path
        self.nir_image_name_list = sorted(os.listdir(self.nir_image_path))
        self.rgb_image_name_list = sorted(os.listdir(self.rgb_image_path))
        self.csv_data = read_csv_file(label_path)
        self.image_count = len(self.nir_image_name_list)
        self.transform = transform

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):

        nir_image_name = self.nir_image_name_list[idx]
        rgb_image_name = self.rgb_image_name_list[idx]

        nir_image_path = os.path.join(self.nir_image_path, nir_image_name)
        rgb_image_path = os.path.join(self.rgb_image_path, rgb_image_name)
        nir_image = cv2.imread(nir_image_path)  # shape [h,w,c]
        rgb_image = cv2.imread(rgb_image_path)  # shape [h,w,c]

        if nir_image.shape[0]!= self.out_h or nir_image.shape[1] != self.out_w:
            nir_image = cv2.resize(nir_image, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        if rgb_image.shape[0]!= self.out_h or rgb_image.shape[1] != self.out_w:
            rgb_image = cv2.resize(rgb_image, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        nir_image = nir_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        nir_image = np.ascontiguousarray(nir_image, dtype=np.float32)  # uint8 to float32
        nir_image = torch.from_numpy(nir_image)

        rgb_image = rgb_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        rgb_image = np.ascontiguousarray(rgb_image, dtype=np.float32)  # uint8 to float32
        rgb_image = torch.from_numpy(rgb_image)

        # if self.paper_affine_generator:
        #     theta = generator_affine_param()
        # else:
        #     theta = random_affine()
        label_row_param = self.csv_data.loc[self.csv_data['image'] == nir_image_name].values
        label_row_param = np.squeeze(label_row_param)
        if nir_image_name != label_row_param[0]:
            raise ValueError("图片文件名和label图片文件名不匹配")

        theta_aff = label_row_param[1:].reshape(2,3)

        theta_aff_tensor = torch.Tensor(theta_aff.astype(np.float32))

        sample = {'nir_image': nir_image, 'rgb_image':rgb_image,'theta': theta_aff_tensor, 'name': nir_image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
使用仿射变换参数生成图片对
返回{"source_image,traget_image,theta_GT,name"}
'''
class NirRgbTnsPair(object):

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
        nir_image_batch,rgb_image_batch,theta_batch,image_name = batch['nir_image'], batch['rgb_image'],batch['theta'],batch['name']
        if self.use_cuda:
            nir_image_batch = nir_image_batch.cuda()
            rgb_image_batch = rgb_image_batch.cuda()
            theta_batch = theta_batch.cuda()

        b, c, h, w = nir_image_batch.size()

        # 为较大的采样区域生成对称填充图像
        rgb_image_batch_pad = symmetricImagePad(rgb_image_batch, self.padding_factor)
        nir_image_batch_pad = symmetricImagePad(nir_image_batch, self.padding_factor)

        # convert to variables 其中Tensor是原始数据，并不知道梯度计算等问题，
        # Variable里面有data，grad和grad_fn，其中data就是Tensor
        # image_batch = Variable(image_batch, requires_grad=False)
        # theta_batch = Variable(theta_batch, requires_grad=False)

        # indices_R = torch.tensor([choice(self.channel_choicelist)])
        # indices_G = torch.tensor([choice(self.channel_choicelist)])

        indices_R = torch.tensor([1])
        indices_G = torch.tensor([0])

        if self.use_cuda:
            indices_R = indices_R.cuda()
            indices_G = indices_G.cuda()

        rgb_image_batch_pad = torch.index_select(rgb_image_batch_pad, 1, indices_R)
        nir_image_batch_pad = torch.index_select(nir_image_batch_pad, 1, indices_G)

        # 获取裁剪的图像
        cropped_image_batch = self.rescalingTnf(rgb_image_batch_pad, None, self.padding_factor,
                                                self.crop_factor)  # Identity is used as no theta given
        # 获取裁剪变换的图像
        warped_image_batch = self.geometricTnf(nir_image_batch_pad, theta_batch,
                                               self.padding_factor,
                                               self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}