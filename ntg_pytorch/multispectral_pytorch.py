
import cv2
import torch
from torch.utils.data import DataLoader
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from datasets.provider.randomTnsData import RandomTnsPair
from datasets.provider.singlechannelData import SingleChannelPairTnf
from datasets.provider.test_dataset import TestDataset
from ntg_pytorch.register_func import estimate_aff_param_iterator, affine_transform
from tnf_transform.img_process import NormalizeImageDict


if __name__ == '__main__':
    use_cuda = False

    img1 = io.imread('datasets/row_data/multispectral/Ir.jpg')
    img2 = io.imread('datasets/row_data/multispectral/Itrot2.jpg')

    # plt.imshow(img1)
    # plt.figure()
    # plt.imshow(img2)
    # plt.show()

    img1 = img1[:, :, 0][:, :, np.newaxis]
    img2 = img2[:, :, 0][:, :, np.newaxis]

    img1 = (img1.astype(np.float32) / 255 - 0.485) / 0.229
    img2 = (img2.astype(np.float32) / 255 - 0.456) / 0.224

    # img1 = torch.Tensor(img1).transpose(1,2).transpose(0,1)
    # img2 = torch.Tensor(img2).transpose(1,2).transpose(0,1)
    #
    # img1 = img1/255.0
    # img2 = img2/255.0

    # source_batch = np.stack((img1,img1),0)
    # target_batch = np.stack((img2,img2),0)

    source_batch = img1[np.newaxis, :]
    target_batch = img2[np.newaxis, :]

    p = estimate_aff_param_iterator(source_batch, target_batch, use_cuda=use_cuda)

    print(p)

    img1 = img1[:, :, 0]
    img2 = img2[:, :, 0]

    im2warped = affine_transform(img2, p[0].numpy())

    [f1x, f1y] = np.gradient(img1)
    [f2x, f2y] = np.gradient(img2)
    [f3x, f3y] = np.gradient(im2warped)
    g1 = np.sqrt(f1x * f1x + f1y * f1y)
    g2 = np.sqrt(f2x * f2x + f2y * f2y)
    g3 = np.sqrt(f3x * f3x + f3y * f3y)

    plt.imshow(img1, cmap='gray')  # 目标图片
    plt.figure()
    plt.imshow(img2, cmap='gray')  # 待变换图片
    plt.figure()
    plt.imshow(im2warped, cmap='gray')
    plt.figure()
    plt.imshow(g1, cmap='gray')
    plt.figure()
    plt.imshow(g2, cmap='gray')
    plt.show()