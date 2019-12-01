from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

class AffineTnf(object):
    def __init__(self,out_h=240,out_w=240,use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        self.gridGen = AffineGridGen(out_h,out_w)
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None,padding_factor=1.0, crop_factor=1.0):
        b, c, h, w = image_batch.size()
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b, 2, 3)  # 扩展维度，添加一个batch size维度
            #theta_batch = Variable(theta_batch, requires_grad=False)

        # 生成采样网格
        sampling_grid = self.gridGen(theta_batch)  # theta_batch [1,2,3]  sampling_grid [1,360,640,2]

        # 根据crop_factor和padding_factor重缩放网格 rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # 采样变换图片 sample transformed image
        warped_image_batch = F.grid_sample(image_batch,
                                           sampling_grid)  # image_batch[1,1,360,640]   warped_image_batch[1,1,360,640]
        return warped_image_batch


class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        self.factor = out_w/out_h

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)

# 通过pytorch的仿射参数生成xy方向的网格点,暂时没用到
def generate_grid(theta,out_h=240,out_w=240):   # theta cuda:0(16,2,3)
    factor = out_w/out_h
    batch_size = theta.size()[0]
    identity_theta = torch.zeros(batch_size,2,3)
    identity_theta += torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
    identity_theta = identity_theta.cuda()

    custom_grid = F.affine_grid(identity_theta, (theta.size()[0], 1, out_h, out_w))
    custom_grid = custom_grid + 1  # (1,339,568,2)
    vector_x = custom_grid[:,:, :, 0].reshape(batch_size,-1) * factor
    vector_y = custom_grid[:,:, :, 1].reshape(batch_size,-1)
    vector_ones = torch.ones(vector_x.size())
    vector_cat = torch.stack((vector_x.double().cuda(), vector_y.double().cuda(), vector_ones.double().cuda()),1).float()

    result = torch.bmm(theta, vector_cat)
    result[:,0, :] = result[:,0, :] / factor - 1
    result[:,1, :] = result[:,1, :] - 1
    result = result.reshape(batch_size,2, out_h, out_w)
    result = result.permute(0, 2, 3, 1).float()
    return result


# 使用pytorch参数的仿射变换
def affine_transform_pytorch(image_batch,theta_batch):
    '''
    :param image_batch: 图片batch Tensor[batch_size,C,240,240]
    :param theta_batch: 参数batch Tensor[batch_size,2,3]
    :return: 变换图片batch warped_image_batch  Tensor[batch_size,C,240,240]
    '''
    theta_batch = theta_batch.reshape(-1,2,3)
    _, _, height,width = image_batch.shape
    gridGen = AffineGridGen(height,width)
    sample_grid = gridGen(theta_batch)
    warped_image_batch = F.grid_sample(image_batch,sample_grid)

    return warped_image_batch

# 使用opencv的仿射变换
def single_affine_transform_opencv(im, p):
    height = im.shape[0]
    width = im.shape[1]
    im = cv2.warpAffine(im,p,(width,height))
    return im

def affine_transform_opencv(image_batch, theta_batch):
    '''
    :param image_batch: Tensor[batch_size,C,240,240]
    :param theta_batch: Tensor[batch_size,2,3]
    :return: warped_img_batch: Tensor[batch_size,240,240]
    '''
    image_batch = image_batch.squeeze(1).detach().cpu().numpy()
    theta_batch = torch.Tensor(theta_batch)
    warped_img_batch = []
    for i in range(len(image_batch)):
        source_img = image_batch[i].squeeze()
        height = source_img.shape[0]
        width = source_img.shape[1]
        warped_img = cv2.warpAffine(source_img, theta_batch[i].numpy(), (width, height))[np.newaxis,:,:]
        warped_img_batch.append(warped_img)

    warped_img_batch = torch.Tensor(warped_img_batch)

    return warped_img_batch