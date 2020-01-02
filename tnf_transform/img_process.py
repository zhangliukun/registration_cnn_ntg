import math
import random

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from tnf_transform.transformation import AffineTnf

# #def random_affine(img= None,degrees=20,translate=.2,scale=.2,shear=10):
# def random_affine(img= None,degrees=30,translate=.3,scale=.3,shear=15):
#     # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#     # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
#
#     # 旋转和缩放
#     R = np.eye(3)
#     a = random.uniform(-degrees,degrees)
#     s = random.uniform(1 - scale, 1 + scale)
#     #R[:2] = cv2.getRotationMatrix2D(angle=a,center=(img.shape[1] / 2, img.shape[0] / 2),scale=s)
#     R[:2] = cv2.getRotationMatrix2D(angle=a,center=(0,0),scale=s)
#
#     # 平移
#     T = np.eye(3)
#     T[0, 2] = random.uniform(-translate, translate)   # x translation (rate)
#     T[1, 2] = random.uniform(-translate, translate)   # y translation (rate)
#
#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
#
#     M = S @ T @ R # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
#
#     return M[0:2]

'''
自定义的仿射变换参数生成
'''
def random_affine(img= None,degrees=5,translate=.05,scale=.05,shear=3,to_dict = False):
#def random_affine(img= None,degrees=20,translate=.2,scale=.2,shear=10,to_dict = False):
#def random_affine(img= None,degrees=30,translate=.3,scale=.3,shear=15,to_dict = False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    # 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees,degrees)
    s = random.uniform(1 - scale, 1 + scale)
    #R[:2] = cv2.getRotationMatrix2D(angle=a,center=(img.shape[1] / 2, img.shape[0] / 2),scale=s)
    R[:2] = cv2.getRotationMatrix2D(angle=a,center=(0,0),scale=s)

    # 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate)   # x translation (rate)
    T[1, 2] = random.uniform(-translate, translate)   # y translation (rate)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    theta = S @ T @ R # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

    theta = theta[0:2]

    if to_dict:
        temp = theta.reshape(6)
        theta = {}
        theta['p0'] = temp[0]
        theta['p1'] = temp[1]
        theta['p2'] = temp[2]
        theta['p3'] = temp[3]
        theta['p4'] = temp[4]
        theta['p5'] = temp[5]

    return theta

# def geometric_random_affine(random_t=0.5, random_s=0.5,random_alpha=1/6):
#
#     alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * random_alpha
#     theta = np.random.rand(6)
#     theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * random_t
#     theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
#     theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
#     theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
#     theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
#     theta = theta.reshape(2, 3)
#
#     return theta

'''
论文中的随机仿射变换参数生成
'''
def generator_affine_param(random_t=0.5,random_s=0.5,random_alpha = 1/6,random_tps=0.4,to_dict = False):
#def generator_affine_param(random_t=0.3,random_s=0.3,random_alpha = 1/8,random_tps=0.4,to_dict = False):
    alpha = (np.random.rand(1)-0.5) * 2 * np.pi * random_alpha
    theta = np.random.rand(6)

    theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * random_t
    theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
    theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
    theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta = theta.reshape(2, 3)

    if to_dict:
        temp = theta.reshape(6)
        theta = {}
        theta['p0'] = temp[0]
        theta['p1'] = temp[1]
        theta['p2'] = temp[2]
        theta['p3'] = temp[3]
        theta['p4'] = temp[4]
        theta['p5'] = temp[5]

    return theta



def preprocess_image(image,resize=True,use_cuda=True):    # image (240,240,3)
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image_var = torch.Tensor(image.astype(np.float32) / 255.0)
    if use_cuda:
        image_var = image_var.cuda()

    # Resize image using bilinear sampling with identity affine tnf
    if resize:
        resizeTnf = AffineTnf(out_h=240, out_w=240, use_cuda = use_cuda)
        image_var = resizeTnf(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var

class NormalizeImage:
    def __init__(self,normalize_range=True,normalize_img = True):
        self.normalize_single_channel = transforms.Normalize(mean=[0.456], std=[0.224])
        self.normalize_rgb_channel = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        self.normalize_range = normalize_range
        self.normalize_img = normalize_img

    def __call__(self, sample):
        if self.normalize_range:
            sample = sample /255.0
        if self.normalize_img:
            sample = self.normalize_single_channel(sample)

        return sample


class NormalizeImageDict:
    """
    Normalizes Tensor images in dictionary

    Args:
        image_keys (list): dict. keys of the images to be normalized
        normalizeRange (bool): if True the image is divided by 255.0s

    """

    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange
        self.normalize_single_channel = transforms.Normalize(mean=[0.456], std=[0.224])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0
            sample[key] = self.normalize_single_channel(sample[key])
        return sample

def normalize_image(image, forward=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    归一化和反归一化图片，forward为True则归一化，否则为反归一化
    :param image:
    :param forward:
    :param mean:
    :param std:
    :return:
    '''

    im_size = image.size()  # torch.Size([1, 3, 240, 240])
    mean = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)  # torch.Size([3, 1, 1])
    std = torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
    if image.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    if isinstance(image, torch.autograd.Variable):
        mean = Variable(mean, requires_grad=False)
        std = Variable(std, requires_grad=False)

    mean = mean[1].unsqueeze(0)
    std = std[1].unsqueeze(0)

    if len(im_size) == 2:
        image = image.unsqueeze(0)
        im_size = image.size()

    if forward:
        if len(im_size) == 3:
            result = image.sub(mean.expand(im_size)).div(std.expand(im_size))
        elif len(im_size) == 4:
            result = image.sub(mean.unsqueeze(0).expand(im_size)).div(std.unsqueeze(0).expand(im_size))
    else:
        if len(im_size) == 3:
            result = image.mul(std.expand(im_size)).add(mean.expand(im_size))
        elif len(im_size) == 4:
            result = image.mul(std.unsqueeze(0).expand(im_size)).add(mean.unsqueeze(0).expand(im_size))


    return result