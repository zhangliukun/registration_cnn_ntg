import time

import cv2
import torch

import matplotlib.pyplot as plt
import numpy as np

from tnf_transform.img_process import random_affine
from tnf_transform.transformation import AffineTnf
from traditional_ntg.estimate_affine_param import estimate_param_batch
from util.pytorchTcv import theta2param, param2theta
from skimage import io

from util.time_util import calculate_diff_time


def evaluate(theta_estimate_batch,theta_GT_batch,source_image_batch,target_image_batch,use_cuda = True):
    # 将pytorch的变换参数转为opencv的变换参数
    theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

    # P5使用传统NTG方法进行优化cnn的结果
    ntg_param = estimate_param_batch(source_image_batch, target_image_batch, None, itermax=600)
    ntg_param_pytorch = param2theta(ntg_param, 240, 240, use_cuda=use_cuda)
    cnn_ntg_param_batch = estimate_param_batch(source_image_batch, target_image_batch, theta_opencv, itermax=800)
    cnn_ntg_param_pytorch_batch = param2theta(cnn_ntg_param_batch, 240, 240, use_cuda=use_cuda)

    # loss_cnn = grid_loss.compute_grid_loss(theta_estimate_batch, theta_GT_batch)
    # loss_ntg = grid_loss.compute_grid_loss(ntg_param_pytorch, theta_GT_batch)
    # loss_cnn_ntg = grid_loss.compute_grid_loss(cnn_ntg_param_pytorch_batch, theta_GT_batch)
    #
    # grid_loss_list.append(loss_cnn)
    # grid_loss_ntg_list.append(loss_ntg)
    # grid_loss_comb_list.append(loss_cnn_ntg)

def compare_img_resize():
    img_path = '/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/multispectral/It.jpg'

    h,w = 600,800

    opencv_start_time = time.time()
    img = cv2.imread(img_path)
    print('imread_time',calculate_diff_time(opencv_start_time))
    img = cv2.resize(img,(w,h),interpolation=cv2.INTER_CUBIC)
    start_time = time.time()
    # img_t = img.transpose(2,0,1)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).unsqueeze(0)

    elpased = calculate_diff_time(opencv_start_time)
    print('opencv time',img.shape,elpased)

    torch_start_time = time.time()
    img2 = io.imread(img_path)
    print('torch_read_time',calculate_diff_time(torch_start_time))
    affineTnf = AffineTnf(h, w, use_cuda=False)
    image = torch.Tensor(img2.astype(np.float32))
    image = image.transpose(1, 2).transpose(0, 1)
    img2 = affineTnf(image.unsqueeze(0))
    elpased = calculate_diff_time(torch_start_time)
    print('torch time,',img2.shape,elpased)

def compare_affine_param_generator(random_t=0.2,random_s=0.2,
                 random_alpha = 1/4):
    start_time = time.time()

    alpha = (torch.rand(1) - 0.5) * 2 * np.pi * random_alpha
    alpha = alpha.numpy()
    theta = torch.rand(6).numpy()

    theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * random_t
    theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
    theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
    theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta = theta.reshape(2, 3)

    elpased = calculate_diff_time(start_time)
    print('计算随机变换参数：', elpased)  # 0.0004s

    start_time = time.time()
    theta_m = random_affine()
    theta_m = torch.Tensor(theta_m)
    elpased = calculate_diff_time(start_time)
    print('随机仿射换换时间:', elpased)

#compare_img_resize()
#compare_affine_param_generator()

