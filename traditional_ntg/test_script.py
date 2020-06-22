import math

import cv2
import torch
from PIL.Image import BILINEAR
from skimage import io
import matplotlib.pyplot as plt

from evluate.lossfunc import GridLoss
from ntg_pytorch.register_func import affine_transform
from ntg_pytorch.register_loss import deriv_filt_pytorch
from tnf_transform.img_process import generate_affine_param
from traditional_ntg.estimate_affine_param import estimate_affine_param
from traditional_ntg.loss_function import deriv_filt
import scipy.misc as smi
import numpy as np
from PIL import Image

# img1 = io.imread('../datasets/row_data/multispectral/mul_1t_s.png')
# img2 = io.imread('../datasets/row_data/multispectral/mul_1s_s.png')

img1 = io.imread('../datasets/row_data/multispectral/fake_and_real_tomatoes_ms_17.png')
img2 = io.imread('../datasets/row_data/multispectral/fake_and_real_tomatoes_ms_28.png')

# img1 = io.imread('../datasets/row_data/texs1.jpeg')
# img2 = io.imread('../datasets/row_data/test2.jpeg')

fn_grid_loss = GridLoss(use_cuda=False,grid_size=512)

center = (256,256)
theta_GT = generate_affine_param(scale=1.1, degree=10, translate_x=-10, translate_y=10, center=center)
theta_GT = torch.from_numpy(theta_GT).unsqueeze(0).float()



# 第一个是target，第二个是source
p = estimate_affine_param(img1,img2,itermax=1000)

im2warped = affine_transform(img2, p)
print(p)
print(theta_GT)

p = torch.from_numpy(p).unsqueeze(0).float()

grid_loss = fn_grid_loss.compute_grid_loss(p,theta_GT)
print(grid_loss)



plt.imshow(img1, cmap='gray')  # 目标图片
plt.figure()
plt.imshow(img2, cmap='gray')  # 待变换图片
plt.figure()
plt.imshow(im2warped, cmap='gray')
plt.show()

# # ration = (1/1.5)**6
# img1 = (img1/255.0)
# #
# # img1 = np.array(Image.fromarray(img1).resize((int(img1.shape[0] * ration), int(img1.shape[1] * ration))))
# #
# # Ix_cv,Iy_cv = deriv_filt(img1,False)
#
# # img1_tensor = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()
# # Ix,Iy = deriv_filt_pytorch(img1_tensor,False,use_cuda=False)
# # Ix = Ix.squeeze().numpy()
# # Iy = Iy.squeeze().numpy()
#
# ration = (1/1.5)
# smooth_sigma = np.sqrt(1.5) / np.sqrt(3)
# kx = cv2.getGaussianKernel(int(2 * round(1.5 * smooth_sigma)) + 1, smooth_sigma)
# ky = cv2.getGaussianKernel(int(2 * round(1.5 * smooth_sigma)) + 1, smooth_sigma)
#
# hg = np.multiply(kx, np.transpose(ky))
# # tmp = cv2.filter2D(img1, -1, hg, borderType=cv2.BORDER_REFLECT)
#
# multi_level_pyramid = []
# for i in range(7):
#     # multi_level_pyramid.append(img1)
#     # Ix_cv, Iy_cv = deriv_filt(img1, False)
#
#     # plt.figure()
#     # plt.imshow(Iy_cv, cmap='gray')
#     if i == 6:
#         plt.figure()
#         plt.imshow(img1, cmap='gray')
#         break
#     # 默认的BORDER_REFLECT_101
#     img1 = cv2.filter2D(img1, -1, hg, borderType=cv2.BORDER_REFLECT_101)
#     img1 = np.array(Image.fromarray(img1).resize((math.ceil(img1.shape[0] * ration), math.ceil(img1.shape[1] * ration)),resample=BILINEAR))
#
#
# # plt.figure()
# # plt.imshow(multi_level_pyramid[0],cmap='gray')
#
# # plt.figure()
# # plt.imshow(Ix,cmap='gray')
# # plt.figure()
# # plt.imshow(Iy,cmap='gray')
# # plt.figure()
# # plt.imshow(Ix_cv,cmap='gray')
# # plt.figure()
# # plt.imshow(Iy_cv,cmap='gray')
# plt.show()

