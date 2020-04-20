import math
import time

import cv2
import numpy as np
import torch

from ntg_pytorch.register_loss import ntg_gradient_torch
from ntg_pytorch.register_pyramid import compute_pyramid, compute_pyramid_pytorch, ScaleTnf, compute_pyramid_iter
import matplotlib.pyplot as plt

from traditional_ntg.loss_function import ntg_gradient


def scale_image(img,IMIN,IMAX):
    return (img-IMIN)/(IMAX-IMIN)

def affine_transform(im,p):
    height = im.shape[0]
    width = im.shape[1]
    im = cv2.warpAffine(im,p,(width,height),flags=cv2.INTER_CUBIC)
    # im = cv2.warpAffine(im,p,(width,height),flags=cv2.INTER_NEAREST)
    return im

'''
注意，如果使用cnn计算出来的参数来给传统方法继续迭代的话，计算高斯金字塔的时候不能进行高斯滤波，因为高斯滤波会降低精度，猜想是因为
有些cnn得到的结果不是很准，这样进行平滑滤波的时候可能会把信息给掩盖掉。
'''
def estimate_aff_param_iterator(source_batch,target_batch,theta_opencv_batch=None,use_cuda=False,itermax = 800,normalize_func = None):

    batch_size = source_batch.shape[0]

    parser = {}
    parser['tol'] = 1e-6
    parser['itermax'] = itermax
    parser['pyramid_spacing'] = 1.5
    parser['minSize'] = 16

    if theta_opencv_batch is None:
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        p = np.tile(p, (batch_size, 1, 1)).astype(np.float32)
        p = torch.from_numpy(p)
        parser['initial_affine_param'] = p
    else:
        parser['initial_affine_param'] = theta_opencv_batch.clone()

    # start_time = time.time()

    pyramid_level1 = 1 + np.floor(np.log(source_batch.shape[2] / parser['minSize']) / np.log(parser['pyramid_spacing']))
    pyramid_level2 = 1 + np.floor(np.log(source_batch.shape[3] / parser['minSize']) / np.log(parser['pyramid_spacing']))
    parser['pyramid_levels'] = np.min((int(pyramid_level1),int(pyramid_level2)))
    # 实测发现如果金字塔不够的话有些情况下可能导致cnn+ntg结合起来的精度还不如传统NTG的精度。
    # print('串行金字塔，层数减1')
    if theta_opencv_batch is not None:
        parser['pyramid_levels'] = parser['pyramid_levels'] -1
    # parser['pyramid_levels'] = 1


    if normalize_func is not None:
        print('分开归一化')
        source_batch = normalize_func.scale_image_batch(source_batch)
        target_batch = normalize_func.scale_image_batch(target_batch)
    else:
        print('原图目标图联合归一化')
        source_batch_max = torch.max(source_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)
        target_batch_max = torch.max(target_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)

        IMAX_index = target_batch_max > source_batch_max
        source_batch_max[IMAX_index] = target_batch_max[IMAX_index]
        IMAX = source_batch_max

        source_batch_min = torch.min(source_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)
        target_batch_min = torch.min(target_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)

        IMIN_index = target_batch_min < source_batch_min
        source_batch_min[IMIN_index] = target_batch_min[IMIN_index]
        IMIN = source_batch_min

        source_batch = scale_image(source_batch,IMIN,IMAX)
        target_batch = scale_image(target_batch,IMIN,IMAX)

    batch_size,channel, h, w = source_batch.shape

    smooth_sigma = np.sqrt(parser['pyramid_spacing']) / np.sqrt(3)
    kx = cv2.getGaussianKernel(int(2 * round(1.5 * smooth_sigma)) + 1, smooth_sigma)
    ky = cv2.getGaussianKernel(int(2 * round(1.5 * smooth_sigma)) + 1, smooth_sigma)
    hg = np.multiply(kx, np.transpose(ky))

    # print("配置时间",calculate_diff_time(start_time))
    # start_time = time.time()

    scaleTnf = ScaleTnf(use_cuda=use_cuda)

    # print('使用pytorch计算金字塔') 实验证明不如串行金字塔好
    # pyramid_images_list = compute_pyramid_pytorch(source_batch,scaleTnf,hg, int(parser['pyramid_levels']),
    #                                       1 / parser['pyramid_spacing'],use_cuda = use_cuda)
    #
    # target_pyramid_images_list = compute_pyramid_pytorch(target_batch,scaleTnf,hg, int(parser['pyramid_levels']),
    #                                              1 / parser['pyramid_spacing'],use_cuda = use_cuda)


    # print('使用不加高斯滤波串行计算金字塔')
    # pyramid_images_list = compute_pyramid(source_batch,hg, int(parser['pyramid_levels']),
    #                                       1 / parser['pyramid_spacing'],use_cuda=use_cuda)
    #
    # target_pyramid_images_list = compute_pyramid(target_batch,hg, int(parser['pyramid_levels']),
    #                                              1 / parser['pyramid_spacing'],use_cuda=use_cuda)
    #
    pyramid_images_list = compute_pyramid_iter(source_batch,hg, int(parser['pyramid_levels']),
                                          1 / parser['pyramid_spacing'],use_cuda=use_cuda)

    target_pyramid_images_list = compute_pyramid_iter(target_batch,hg, int(parser['pyramid_levels']),
                                                 1 / parser['pyramid_spacing'],use_cuda=use_cuda)

    # plt.show()

    # print("pytorch金字塔",calculate_diff_time(start_time))
    # start_time = time.time()

    # 这里因为传入的变换参数是从CNN中获得的，大小为240*240，所以传入进来使用的话需要进行缩放，使用最小层除以最大层得到缩放比例，
    # 然后就得到CNN变换比例得到的最小层相应的大小了。
    if theta_opencv_batch is not None:
        ration_diff = pyramid_images_list[-1].shape[-1] / pyramid_images_list[0].shape[-1]
        parser['initial_affine_param'][:,0, 2] = parser['initial_affine_param'][:,0, 2]*ration_diff
        parser['initial_affine_param'][:,1, 2] = parser['initial_affine_param'][:,1, 2]*ration_diff

    for k in range(parser['pyramid_levels'] - 1, -1, -1):
        if k == (parser['pyramid_levels'] - 1):
            p = parser['initial_affine_param']
            if use_cuda:
                p = p.cuda()

        else:
            parser['itermax'] = math.ceil(parser['itermax'] / parser['pyramid_spacing'])
            p[:, 0, 2] = p[:, 0, 2] * pyramid_images_list[k].shape[3] / pyramid_images_list[k + 1].shape[3]
            p[:, 1, 2] = p[:, 1, 2] * pyramid_images_list[k].shape[2] / pyramid_images_list[k + 1].shape[2]

        copy = {}
        copy['parser'] = parser
        # copy['source_images'] = torch.from_numpy(pyramid_images_list[k]).float()
        # copy['target_images'] = torch.from_numpy(target_pyramid_images_list[k]).float()
        copy['source_images'] = pyramid_images_list[k]
        copy['target_images'] = target_pyramid_images_list[k]

        if use_cuda:
            copy['source_images'] = copy['source_images'].cuda()
            copy['target_images'] = copy['target_images'].cuda()

        sz = [pyramid_images_list[k].shape[2], pyramid_images_list[k].shape[3]]
        xlist = torch.tensor(range(0, sz[1]))
        ylist = torch.tensor(range(0, sz[0]))

        if use_cuda:
            xlist = xlist.cuda()
            ylist = ylist.cuda()

        [X_array, Y_array] = torch.meshgrid(xlist, ylist)

        X_array = X_array.float().transpose(0, 1)
        Y_array = Y_array.float().transpose(0, 1)

        # copy['W_array'] = X_array.expand(batch_size, channel, -1, -1)
        # copy['H_array'] = Y_array.expand(batch_size, channel, -1, -1)

        copy['X_array'] = (X_array / torch.max(X_array)).expand(batch_size,channel,-1,-1)
        copy['Y_array'] = (Y_array / torch.max(Y_array)).expand(batch_size,channel,-1,-1)

        converged = False
        iter = 0
        steplength = 0.5 / np.max(sz)

        while not converged:
            start_time = time.time()

            # source_image_batch = copy['source_images'].squeeze().numpy()
            # target_image_batch = copy['target_images'].squeeze().numpy()
            # images = np.stack((source_image_batch,target_image_batch),2)
            # copy['images'] =images
            # copy['options'] = None
            # copy['X'] = (X_array / torch.max(X_array)).expand(batch_size, channel, -1, -1).numpy()
            # copy['Y'] = (Y_array / torch.max(Y_array)).expand(batch_size, channel, -1, -1).numpy()
            # g = ntg_gradient(copy,p.squeeze().numpy())
            # g = torch.from_numpy(g).unsqueeze(0).float()

            g = ntg_gradient_torch(copy, p, use_cuda=use_cuda).detach()
            # print("ntg_gradient_torch", calculate_diff_time(start_time))
            if p is None:
                print("p is None")
            p = p + steplength * g / torch.max(torch.abs(g+1e-16).view(g.shape[0],-1),1)[0].unsqueeze(1).unsqueeze(1)
            #residualError = torch.max(torch.abs(g[0]))
            residualError = torch.max(torch.abs(g).view(g.shape[0],-1),1)[0]
            iter = iter + 1
            #converged = (iter >= parser['itermax']) or (residualError < parser['tol'])
            converged = iter >= parser['itermax']
            # print(converged)
            # if converged:
            #     print(str(k) + " " + str(iter) + " " + str(residualError[0:8]))
                #print(str(k) + " " + str(iter))
                #torch.cuda.empty_cache()

    #print("循环结束时间：",calculate_diff_time(start_time))

    return p

