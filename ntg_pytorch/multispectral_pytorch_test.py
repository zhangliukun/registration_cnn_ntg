
import cv2
import torch
from torch.utils.data import DataLoader
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from datasets.provider.randomTnsData import RandomTnsPair
from datasets.provider.singlechannelData import SingleChannelPairTnf
from datasets.provider.test_dataset import TestDataset
from ntg_pytorch.register_func import estimate_aff_param_iterator, affine_transform, scale_image
from tnf_transform.img_process import NormalizeImageDict, generate_affine_param
from traditional_ntg.estimate_affine_param import estimate_affine_param
from visualization.train_visual import VisdomHelper

def use_torch_ntg(img1,img2):
    img1 = img1[np.newaxis, np.newaxis, :, :]
    img2 = img2[np.newaxis, np.newaxis, :, :]

    source_batch = torch.from_numpy(img1).float()
    target_batch = torch.from_numpy(img2).float()

    p = estimate_aff_param_iterator(source_batch, target_batch, use_cuda=use_cuda, itermax=600)
    p = p[0].cpu().numpy()
    return p

def use_cv2_ntg(img1,img2):
    p = estimate_affine_param(img2,img1,itermax=600)
    return p


if __name__ == '__main__':

    small = True
    if small:
        theta = generate_affine_param(scale=1.1, degree=10, translate_x=-10, translate_y=10)
    else:
        theta = generate_affine_param(scale=1.25, degree=30, translate_x=-20, translate_y=20)

    use_cuda = torch.cuda.is_available()

    env = "ntg_pytorch"
    vis = VisdomHelper(env)

    img1 = io.imread('../datasets/row_data/multispectral/mul_1s_s.png')
    img2 = io.imread('../datasets/row_data/multispectral/mul_1t_s.png')

    p = use_torch_ntg(img1,img2)

    print(p)

    im2warped = affine_transform(img1,p)
    imgGt = affine_transform(img1,theta)

    plt.imshow(img1, cmap='gray')  # 目标图片
    plt.figure()
    plt.imshow(img2, cmap='gray')  # 待变换图片
    plt.figure()
    plt.imshow(im2warped, cmap='gray')
    plt.figure()
    plt.imshow(imgGt, cmap='gray')
    plt.show()