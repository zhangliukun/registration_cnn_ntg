
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
from evluate.lossfunc import GridLoss
from ntg_pytorch.register_func import estimate_aff_param_iterator, affine_transform, scale_image
from tnf_transform.img_process import NormalizeImageDict, generate_affine_param, NormalizeCAVEDict
from traditional_ntg.estimate_affine_param import estimate_affine_param
from util.pytorchTcv import inverse_theta, param2theta
from visualization.train_visual import VisdomHelper
from sklearn import metrics

def use_torch_ntg(img1,img2):
    img1 = img1[np.newaxis, np.newaxis, :, :]
    img2 = img2[np.newaxis, np.newaxis, :, :]

    source_batch = torch.from_numpy(img1).float()
    target_batch = torch.from_numpy(img2).float()

    # normalize_func = NormalizeCAVEDict(["image"])
    p = estimate_aff_param_iterator(source_batch, target_batch, use_cuda=use_cuda, itermax=600)
    p = p[0].cpu().numpy()
    return p

def use_cv2_ntg(img1,img2):
    p = estimate_affine_param(img2,img1,itermax=600)
    return p


if __name__ == '__main__':

    small = True

    # img1 = io.imread('../datasets/row_data/multispectral/fake_and_real_tomatoes_ms_31.png') * 1.0
    # img2 = io.imread('../datasets/row_data/multispectral/fake_and_real_tomatoes_ms_17.png') * 1.0

    img1 = io.imread('../datasets/row_data/multispectral/mul_1s_s.png')
    img2 = io.imread('../datasets/row_data/multispectral/mul_1t_s.png')

    center = (img1.shape[0]/2,img1.shape[1]/2)
    center = (0,0)
    if small:
        theta = generate_affine_param(scale=1.1, degree=10, translate_x=-10, translate_y=10,center=center)
    else:
        theta = generate_affine_param(scale=1.25, degree=30, translate_x=-20, translate_y=20,center=center)

    use_cuda = torch.cuda.is_available()

    env = "ntg_pytorch"
    vis = VisdomHelper(env)

    p = use_torch_ntg(img1,img2)
    # p = use_cv2_ntg(img1,img2)


    # im2warped = affine_transform(img2,p)
    im2warped = affine_transform(img1,p)

    imgGt = affine_transform(img1,theta)

    print(metrics.normalized_mutual_info_score(im2warped.flatten()/255.0, imgGt.flatten()/255.0))


    p = torch.from_numpy(p).unsqueeze(0).float()

    p_pytorch = param2theta(p,img1.shape[0],img1.shape[1],use_cuda=False)

    theta_GT = torch.from_numpy(theta).unsqueeze(0).float()
    theta_GT = param2theta(theta_GT,img1.shape[0],img1.shape[1],use_cuda=False)
    fn_grid_loss = GridLoss(use_cuda=False, grid_size=512)


    print(p_pytorch)
    print(theta_GT)

    grid_loss = fn_grid_loss.compute_grid_loss(p_pytorch, theta_GT)
    print('grid_loss',grid_loss)


    plt.imshow(img1, cmap='gray')  # 目标图片
    plt.figure()
    plt.imshow(img2, cmap='gray')  # 待变换图片
    plt.figure()
    plt.imshow(im2warped, cmap='gray')
    plt.figure()
    plt.imshow(imgGt, cmap='gray')
    plt.show()