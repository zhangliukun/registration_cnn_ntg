import math

import cv2
import numpy as np
import torch

from ntg_pytorch.register_loss import ntg_gradient_torch
from ntg_pytorch.register_pyramid import compute_pyramid, compute_pyramid_pytorch, ScaleTnf


def scale_image(img,IMIN,IMAX):
    return (img-IMIN)/(IMAX-IMIN)

def affine_transform(im,p):
    height = im.shape[0]
    width = im.shape[1]
    im = cv2.warpAffine(im,p,(width,height))
    return im

def estimate_aff_param_iterator(source_batch,target_batch,use_cuda=False):

    parser = {}
    parser['tol'] = 1e-6
    parser['itermax'] = 500
    parser['pyramid_spacing'] = 1.5
    parser['minSize'] = 16
    parser['initial_affine_param'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    parser['pyramid_levels'] = 8

    batch_size = source_batch.shape[0]

    scaleTnf = ScaleTnf()

    source_batch_max = torch.max(source_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)
    target_batch_max = torch.max(target_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)

    source_batch_min = torch.min(source_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)
    target_batch_min = torch.min(source_batch.view(batch_size,1,-1),2)[0].unsqueeze(2).unsqueeze(2)

    source_batch = scale_image(source_batch,source_batch_min,source_batch_max)
    target_batch = scale_image(target_batch,target_batch_min,target_batch_max)

    smooth_sigma = np.sqrt(parser['pyramid_spacing']) / np.sqrt(3)
    kx = cv2.getGaussianKernel(int(2 * round(1.5 * smooth_sigma)) + 1, smooth_sigma)
    ky = cv2.getGaussianKernel(int(2 * round(1.5 * smooth_sigma)) + 1, smooth_sigma)
    hg = np.multiply(kx, np.transpose(ky))

    # pyramid_images_list = compute_pyramid(source_batch,hg, int(parser['pyramid_levels']),
    #                                       1 / parser['pyramid_spacing'])
    # target_pyramid_images_list = compute_pyramid(target_batch,hg, int(parser['pyramid_levels']),
    #                                              1 / parser['pyramid_spacing'])

    pyramid_images_list = compute_pyramid_pytorch(source_batch,scaleTnf,hg, int(parser['pyramid_levels']),
                                          1 / parser['pyramid_spacing'])

    target_pyramid_images_list = compute_pyramid_pytorch(target_batch,scaleTnf,hg, int(parser['pyramid_levels']),
                                                 1 / parser['pyramid_spacing'])


    for k in range(parser['pyramid_levels'] - 1, -1, -1):
        if k == (parser['pyramid_levels'] - 1):
            p = parser['initial_affine_param']
            p = np.tile(p, (batch_size, 1, 1)).astype(np.float32)
            p = torch.from_numpy(p)
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

        copy['X_array'] = X_array / torch.max(X_array)
        copy['Y_array'] = Y_array / torch.max(Y_array)

        converged = False
        iter = 0
        steplength = 0.5 / np.max(sz)

        while not converged:
            g = ntg_gradient_torch(copy, p, use_cuda=use_cuda)
            if p is None:
                print("p is None")
            p = p + steplength * g / torch.max(torch.abs(g))
            residualError = torch.max(torch.abs(g))
            iter = iter + 1
            converged = (iter >= parser['itermax']) or (residualError < parser['tol'])
            # print(converged)
            if converged:
                print(str(k) + " " + str(iter) + " " + str(residualError))

    return p

