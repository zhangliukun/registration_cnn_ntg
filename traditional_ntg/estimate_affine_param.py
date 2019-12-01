import time

import cv2
import numpy as np
import torch
import traditional_ntg.image_util as imgUtil
import traditional_ntg.compute_image_pyramid as pyramid
import math
import traditional_ntg.loss_function as lossfn
import matplotlib.pyplot as plt
from skimage import io

# 传统方法的包装方法，批量运算
from util.time_util import calculate_diff_time

# 使用传统方法进行优化
def estimate_param_batch(source_image_batch,target_image_batch,theta_opencv_batch=None,itermax = 500):
    '''
    :param source_image_batch: Tensor[batch_size,C,h,w]
    :param target_image_batch: Tensor[batch_size,C,h,w]
    :param theta_opencv_batch: Tensor[batch_size,3,3]
    :return: opencv变换参数
    '''
    ntg_param_batch = []

    for i in range(len(source_image_batch)):
        if theta_opencv_batch is None:
            ntg_param = estimate_affine_param(target_image_batch[i].squeeze().detach().cpu().numpy(),
                                              source_image_batch[i].squeeze().detach().cpu().numpy(),
                                              None, itermax=itermax)
        else:
            ntg_param = estimate_affine_param(target_image_batch[i].squeeze().detach().cpu().numpy(),
                                              source_image_batch[i].squeeze().detach().cpu().numpy(),
                                              theta_opencv_batch[i].detach().cpu().numpy(), itermax=itermax)

        ntg_param_batch.append(ntg_param)

    ntg_param_batch = torch.Tensor(ntg_param_batch)
    return ntg_param_batch

# 单个传统方法的运算
# img1是target_image, img2是source_image
def estimate_affine_param(img1,img2,p=None,itermax = 300):
    '''
    :param img1: img1是target_image
    :param img2: img2是source_image
    :param p: p为None则初始化为单位矩阵，不为None则继承运行
    :param itermax:
    :return: 返回opencv的参数
    '''

    options = {}
    options['tol'] = 1e-6
    options['itermax'] = itermax
    #options['itermax'] = 100
    options['minSize'] = 16
    options['pyramid_spacing'] = 1.5
    options['display'] = True
    options['deriv_filter'] = np.array([-0.5, 0, 0.5])
    options['deriv_filter_conj'] = np.array([0.5, 0, -0.5])
    if p is None:
        options['initial_affine_param'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    else:
        options['initial_affine_param'] = np.copy(p)
        #options['initial_affine_param'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pyramid_level1 = 1 + np.floor(np.log(img1.shape[0] / options["minSize"]) / np.log(options["pyramid_spacing"]));
    pyramid_level2 = 1 + np.floor(np.log(img1.shape[1] / options["minSize"]) / np.log(options["pyramid_spacing"]));
    options['pyramid_levels'] = np.min((int(pyramid_level1),int(pyramid_level2)));
    #options['pyramid_levels'] = 6

    IMAX = np.max([np.max(img1),np.max(img2)])
    IMIN = np.min([np.min(img1),np.min(img2)])

    images1 = imgUtil.scale_image(img1,IMIN,IMAX)
    images2 = imgUtil.scale_image(img2,IMIN,IMAX)

    images = np.stack((images1,images2),axis=2)
    smooth_sigma = np.sqrt(options['pyramid_spacing']) / np.sqrt(3);

    # plt.figure()
    # plt.imshow(img1,cmap='gray')
    # plt.figure()
    # plt.imshow(images1,cmap=plt.cm.gray_r)
    # plt.show()

    kx = cv2.getGaussianKernel(int(2*round(1.5*smooth_sigma))+1,smooth_sigma)
    ky = cv2.getGaussianKernel(int(2*round(1.5*smooth_sigma))+1,smooth_sigma)
    hg = np.multiply(kx,np.transpose(ky))

    start_time = time.time()
    pyramid_images = pyramid.compute_image_pyramid(images,hg,int(options['pyramid_levels']),1/options['pyramid_spacing'])
    elpased = calculate_diff_time(start_time)
    #print('计算图像金字塔时间:',elpased)

    options['initial_affine_param'][0, 2] = options['initial_affine_param'][0, 2]/240 * pyramid_images[-1].shape[0]
    options['initial_affine_param'][1, 2] = options['initial_affine_param'][1, 2]/240 * pyramid_images[-1].shape[0]

    start_time = time.time()
    for k in range(options['pyramid_levels']-1,-1,-1):
        if k == (options['pyramid_levels']-1):
            p = options['initial_affine_param']
            #p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            options['itermax'] = math.ceil(options['itermax']/options['pyramid_spacing'])
            p[0,2] = p[0,2] *pyramid_images[k].shape[1]/pyramid_images[k+1].shape[1]
            p[1,2] = p[1,2] *pyramid_images[k].shape[0]/pyramid_images[k+1].shape[0]

        # 生成当前层设置的拷贝
        small = {}
        small['options'] = options
        small['images'] = pyramid_images[k]

        # 在当前层估计仿射变换参数
        sz = [pyramid_images[k].shape[0],pyramid_images[k].shape[1]]
        xlist = range(0,sz[1])
        ylist = range(0,sz[0])

        [X,Y] = np.meshgrid(xlist,ylist)

        small['X'] = X/np.max(X)
        small['Y'] = Y/np.max(Y)

        converged = False
        iter = 0
        #steplength = 0.5/np.max(sz)
        steplength = 0.5/np.max(sz)

        while not converged:
            g = lossfn.ntg_gradient(small,p)
            if p is None:
                print("p is None")
            p = p + steplength*g/np.max(np.abs(g+1e-16))
            residualError = np.max(np.abs(g))
            iter = iter + 1
            converged = (iter>=options['itermax']) or (residualError < options['tol'])
            #print(converged)
            # if converged:
            #     print(str(k)+" "+str(iter)+" "+ str(residualError))

    elpased = calculate_diff_time(start_time)
    #print("迭代优化时间：",elpased)

    return p

